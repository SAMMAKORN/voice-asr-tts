#!/usr/bin/env python3
"""คุยโต้ตอบกับ AI ด้วยเสียงแบบเรียลไทม์ (ไทย)

  เสียงเข้า → ASR → LLM (สตรีม) → TTS → เสียงออก
  พูดแทรกได้ตลอดเวลา · แสดง caption ทันที · บันทึกบทสนทนาลง logs/

ใช้ค่า API/โมเดลจาก .env ทั้งหมด
"""
from __future__ import annotations

import argparse
import collections
import os
import queue
import re
import sys
import threading
import time
from typing import Callable

import numpy as np

from vc.api import ApiClient, ApiError
from vc.audio import Microphone, Speaker, list_devices, mac_input_volume
from vc.chunker import SentenceChunker, clean_for_tts, split_for_tts
from vc.config import Config, load_config
from vc.logger import SessionLogger
from vc.tools import TOOLS, ToolRunner, describe, wrap_external
from vc.ui import Console, CYAN, GREEN, GRAY, MAGENTA, YELLOW
from vc.vad import VoiceGate

GREETING = "สวัสดีครับ ผมพร้อมคุยแล้ว พูดได้เลยครับ พูดแทรกได้ตลอดเวลา"
SEARCH_FILLER = "ขอค้นข้อมูลสักครู่นะครับ"
CUT_MARK = " …(ผู้ใช้พูดแทรกตรงนี้ ส่วนท้ายอาจยังไม่ได้ยิน)"
NO_ANSWER_MARK = "…(ผู้ใช้พูดแทรกก่อนที่จะได้ตอบ)"
FINDINGS_HEADER = (
    "ข้อมูลที่คุณค้นเจอไปแล้วก่อนหน้านี้ในบทสนทนาเดียวกัน "
    "ใช้ตอบต่อได้เลยโดยไม่ต้องค้นซ้ำ ถ้าผู้ใช้ถามย้ำเรื่องเดิม:\n\n")
THAI_CHARS = re.compile(r"[฀-๿]")
LOW_INPUT_VOLUME = 45      # ต่ำกว่านี้บน macOS ถือว่าไมค์ถูกหรี่จนใช้งานไม่ได้
# เพดานเวลารอให้ TTS พูดจบต่อหนึ่งเทิร์น — กันลูปรอวนไม่จบถ้ามีอะไรค้างผิดปกติ
# (คำตอบยาวสุดที่เจอในบันทึกจริงใช้เวลาราว 20 วินาที จึงเผื่อไว้ 30)
TTS_WAIT_TIMEOUT = 30.0
TTS_WAIT_POLL = 0.03


class VoiceChat:
    def __init__(self, cfg: Config, args: argparse.Namespace, console: Console | None = None):
        self.cfg = cfg
        self.args = args
        self.console = console or Console()   # โหมดเว็บส่ง WebConsole เข้ามาแทน
        self.api = ApiClient(cfg)
        self.log = SessionLogger(
            cfg.log_dir,
            save_audio=cfg.save_audio,
            meta={
                "chat_model": cfg.chat_model,
                "asr_model": cfg.asr_model,
                "tts_model": cfg.tts_model,
                "base_url": cfg.base_url,
            },
        )

        self.tools = ToolRunner(cfg) if cfg.web_search else None
        self.messages: list[dict] = [cfg.system_message()]
        # ผลค้นเว็บของเทิร์นก่อน ๆ — เก็บแยกจาก messages เพราะต้องรอดจากการตัดประวัติ
        self.findings: collections.deque[str] = collections.deque(maxlen=cfg.keep_findings)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._state_lock = threading.Lock()
        self.epoch = 0
        self.cancel = threading.Event()
        self.busy = False               # กำลังคิด/พูด → พูดแทรกได้
        self.utterance_no = 0
        self._barged = False            # เทิร์นล่าสุดถูกตัดเพราะ VAD จับว่ามีคนพูด
        self._last_reply: dict | None = None   # ไว้กู้คืนถ้าที่ตัดไปเป็นเสียงหลอน
        self.audio_stuck = False        # ปิด stream ไม่ลง (ดู vc.audio.close_stream)

        self.tts_queue: queue.Queue[tuple[int, int, str] | None] = queue.Queue()
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._wait_spins = 0            # นับรอบของลูปรอ TTS (ใช้ยืนยันว่าไม่มี busy-loop)
        self._close_lock = threading.Lock()
        self._resources_closed = False

        self.mic: Microphone | None = None
        self.speaker: Speaker | None = None
        self.gate: VoiceGate | None = None
        self.running = threading.Event()
        self.running.set()

    # ---------------------------------------------------------------- ระบบเสียง
    def start_audio(self) -> None:
        self.speaker = Speaker(self.cfg.speaker_sr, self.cfg.output_device)
        out_sr = self.speaker.start()
        if out_sr != self.cfg.speaker_sr:
            self.console.note(f"  ลำโพงทำงานที่ {out_sr} Hz → แปลงเสียงจาก "
                              f"{self.cfg.speaker_sr} Hz ให้อัตโนมัติ")
        if self.args.no_mic:
            return
        self.mic = Microphone(self.cfg.mic_sr, self.cfg.frame_ms, self.cfg.input_device)
        native = self.mic.start()
        if native != self.cfg.mic_sr:
            self.console.note(f"  ไมค์ทำงานที่ {native} Hz → แปลงเป็น {self.cfg.mic_sr} Hz")
        self.gate = VoiceGate(
            self.cfg, self.mic, self.speaker,
            on_speech_start=self._on_speech_start,
            on_utterance=self._on_utterance,
        )
        self.console.status("🎚", "กำลังวัดเสียงรบกวนรอบข้าง อยู่เงียบ ๆ ครู่หนึ่ง...")
        noise = self.gate.calibrate(1.2)
        gain = self.apply_mic_gain(noise)
        self.console.clear_status()
        self.console.note(f"  ระดับเสียงรบกวน {self.gate.noise:.4f} · "
                          f"เกณฑ์เริ่มอัด {max(self.cfg.vad_abs_threshold, self.gate.noise * self.cfg.vad_noise_mult):.4f}"
                          + (f" · ขยายเสียงไมค์ {gain:.1f} เท่า" if gain > 1.05 else ""))
        self.warn_if_mic_quiet(noise)
        self.log.event("calibrated", noise=round(noise, 5), gain=round(gain, 2))
        self.gate.start()

    def apply_mic_gain(self, noise: float) -> float:
        """ชดเชย gain ขาเข้าที่ต่ำเกินไป (แทน autoGainControl ที่มีแต่ฝั่งเบราว์เซอร์)

        เกณฑ์ของ VAD ตั้งไว้กับระดับเสียงพูด "ปกติ" ถ้า input volume ของเครื่อง
        ถูกหรี่ไว้ เสียงพูดจริงจะต่ำกว่าเกณฑ์ตลอดและระบบจะเงียบสนิทเหมือนไมค์เสีย
        จึงขยายให้พื้นเสียงรบกวนกลับมาอยู่ระดับที่เกณฑ์เดิมใช้ได้
        """
        if self.mic is None or noise <= 0:
            return 1.0
        # โหมดอัตโนมัติจำกัดไว้ 8 เท่า กันกรณีห้องเงียบผิดปกติแล้วขยายจนเสียงพูดคลิป
        # (ใส่ MIC_GAIN เองได้ถึง 20 เท่า)
        gain = self.cfg.mic_gain or min(8.0, self.cfg.mic_target_noise / noise)
        gain = self.mic.set_gain(gain)
        if self.gate is not None:
            self.gate.noise = noise * gain      # ค่าที่วัดไว้ก่อนขยาย ต้องสเกลตาม
        return gain

    def warn_if_mic_quiet(self, noise: float) -> None:
        vol = mac_input_volume()
        if vol is not None and vol < LOW_INPUT_VOLUME:
            self.console.warn(
                f"  ระดับเสียงเข้า (input volume) ของเครื่องอยู่ที่ {vol}% — ต่ำมาก")
            self.console.note("  แก้ที่ System Settings › Sound › Input หรือสั่ง:")
            self.console.note("      osascript -e 'set volume input volume 80'")
            self.log.event("low_input_volume", percent=vol)

    # -------------------------------------------------------- callback จาก VAD
    def _on_speech_start(self) -> None:
        """ผู้ใช้เริ่มพูด — ถ้า AI กำลังคิดหรือพูดอยู่ ให้หยุดทันที"""
        with self._state_lock:
            busy = self.busy
        if busy:
            self._barged = True
            self.interrupt("ผู้ใช้พูดแทรก")

    def _on_utterance(self, pcm: np.ndarray) -> None:
        self.events.put(("utterance", pcm))

    def interrupt(self, reason: str, log_event: bool = True) -> None:
        self.cancel.set()
        if self.speaker is not None:
            self.speaker.stop()
        self._drain_tts_queue()
        if log_event:
            self.log.event("interrupt", reason=reason, epoch=self.epoch)

    def _drain_tts_queue(self) -> None:
        while True:
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._dec_inflight()

    # ------------------------------------------------------------------- TTS
    def _inc_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight += 1

    def _dec_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def _tts_busy(self) -> bool:
        with self._inflight_lock:
            return self._inflight > 0

    def _tts_worker(self) -> None:
        assert self.speaker is not None
        try:
            self._tts_loop()
        finally:
            # ออกจากลูปเมื่อไรก็ตาม ต้องคืนตัวนับให้ครบทุก item ที่ค้างอยู่
            # ไม่งั้น `_tts_busy()` ค้างเป็น True ตลอดกาลและลูปรอจะไม่มีวันจบ
            # invariant: หลังเธรดนี้จบ inflight == 0 และคิวว่างเสมอ
            self._drain_tts_queue()

    def _tts_loop(self) -> None:
        assert self.speaker is not None
        while self.running.is_set():
            try:
                item = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            epoch, seq, text = item
            try:
                if epoch != self.epoch or self.cancel.is_set():
                    continue
                spoken = clean_for_tts(text)
                if not spoken:
                    continue
                t0 = time.perf_counter()
                pcm = self.api.synthesize(spoken, self.cfg.speaker_sr)
                if epoch != self.epoch or self.cancel.is_set():
                    continue
                self.speaker.play(pcm, tag=(epoch, seq, text))
                self.log.event(
                    "tts", epoch=epoch, seq=seq, chars=len(spoken),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    audio_ms=int(pcm.size / self.cfg.speaker_sr * 1000),
                )
            except ApiError as exc:
                self.console.warn(f"TTS ล้มเหลว: {exc}")
                self.log.event("tts_error", epoch=epoch, seq=seq, error=str(exc))
            except Exception as exc:  # noqa: BLE001
                self.log.event("tts_error", epoch=epoch, seq=seq, error=repr(exc))
            finally:
                self._dec_inflight()

    def _tts_pending(self) -> bool:
        return self._tts_busy() or bool(self.speaker is not None and self.speaker.pending())

    def wait_for_tts(self, cancel: threading.Event | None = None,
                     on_tick: Callable[[], None] | None = None) -> bool:
        """รอจนพูดจบ — คืน True ถ้าพูดจบเองตามปกติ

        ทุกเงื่อนไขที่ทำให้ต้องเลิกรอถูกรวมไว้ที่เดียว: ถูกพูดแทรก (`cancel`),
        ระบบกำลังปิด (`running`) หรือรอเกินเพดานเวลา ก่อนหน้านี้ลูปนี้เช็คแค่
        `cancel` ซึ่งไม่มีวันเป็น True ตอนปิด session → เธรดค้างและวนกินซีพียู
        """
        deadline = time.monotonic() + TTS_WAIT_TIMEOUT
        while self._tts_pending():
            if not self.running.is_set():
                return False
            if cancel is not None and cancel.is_set():
                return False
            if time.monotonic() > deadline:
                self.console.warn(f"  รอเสียงพูดจบเกิน {TTS_WAIT_TIMEOUT:.0f} วินาที "
                                  "— ข้ามไปก่อน")
                self.log.event("tts_wait_timeout", seconds=TTS_WAIT_TIMEOUT)
                return False
            self._wait_spins += 1
            if on_tick is not None:
                on_tick()
            time.sleep(TTS_WAIT_POLL)
        return True

    def enqueue_tts(self, epoch: int, seq: int, text: str) -> None:
        if not self.cfg.tts_enabled:
            return
        self._inc_inflight()
        self.tts_queue.put((epoch, seq, text))

    # ------------------------------------------------------------------ stdin
    def _stdin_worker(self) -> None:
        for line in sys.stdin:
            if not self.running.is_set():
                return
            text = line.strip()
            if text == "":
                if self.busy:
                    self.interrupt("กด Enter")
                continue
            low = text.lower()
            if low in ("/q", "/quit", "/exit", "ออก"):
                self.events.put(("quit", None))
                return
            if low in ("/mute", "/m"):
                self.cfg.tts_enabled = not self.cfg.tts_enabled
                self.console.note(f"  เสียงพูดของ AI: "
                                  f"{'เปิด' if self.cfg.tts_enabled else 'ปิด'}")
                continue
            if low in ("/clear", "/reset"):
                self.reset_history()
                self.console.note("  ล้างประวัติการสนทนาแล้ว")
                self.log.event("history_cleared")
                continue
            if low in ("/log", "/logs"):
                self.console.note(f"  บันทึกอยู่ที่ {self.log.dir}")
                continue
            if low in ("/help", "/h", "/?"):
                self.print_help()
                continue
            self.events.put(("text", text))

    # ------------------------------------------------------------------- turn
    def _new_epoch(self) -> int:
        with self._state_lock:
            self.epoch += 1
            self.cancel = threading.Event()
            self.busy = True
            return self.epoch

    def _finish_epoch(self) -> None:
        with self._state_lock:
            self.busy = False

    def _is_echo_noise(self, text: str, barged: bool) -> bool:
        """ตัดสินว่าเสียงที่เพิ่งจับได้เป็นคนพูดจริง หรือเสียงลำโพงตัวเองย้อนเข้าไมค์

        ตอน AI กำลังพูด เสียงของมันเองรั่วเข้าไมค์เป็นช่วงสั้น ๆ แล้ว ASR จะ
        "เดา" ออกมาเป็นข้อความมั่ว ๆ มักเป็นภาษาอื่นและสั้น (เจอจริงในบันทึก:
        啥东西？ / 我爱你。 / bản thân cậu.) ถ้าเชื่อตามนั้นจะได้ผลสองต่อ คือ
        คำตอบจริงถูกทิ้ง และประวัติสนทนาถูกยัดขยะจนโมเดลตามเรื่องไม่ทัน
        """
        t = text.strip()
        if len(t) < self.cfg.barge_in_min_chars:
            return True
        if (barged and len(t) < 40 and self.cfg.lang_hint.startswith("th")
                and not THAI_CHARS.search(t)):
            return True
        return False

    def _repair_last_reply(self) -> None:
        """คืนคำตอบเต็มให้ประวัติ หลังพบว่าที่ตัดไปไม่ใช่เสียงคนพูดจริง"""
        last = self._last_reply
        if not last or not last["full"]:
            return
        idx = last["idx"]
        if 0 <= idx < len(self.messages) and self.messages[idx] is last["message"]:
            self.messages[idx]["content"] = last["full"]
            self.log.event("reply_restored", chars=len(last["full"]))
        self._last_reply = None

    def handle_utterance(self, first: np.ndarray) -> None:
        epoch = self._new_epoch()
        cancel = self.cancel
        barged, self._barged = self._barged, False
        pcms = [first]
        parts: list[str] = []
        try:
            while True:
                self.console.status("📝", "กำลังถอดเสียง...", MAGENTA)
                for pcm in pcms:
                    self.utterance_no += 1
                    t0 = time.perf_counter()
                    text = self.api.transcribe(pcm, self.cfg.mic_sr)
                    audio_path = self.log.save_utterance(
                        pcm, self.cfg.mic_sr, self.utterance_no)
                    self.log.event(
                        "asr", epoch=epoch, text=text,
                        audio_ms=int(pcm.size / self.cfg.mic_sr * 1000),
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                        audio_file=audio_path,
                    )
                    if text:
                        parts.append(text)
                # ถ้าผู้ใช้พูดต่อระหว่างถอดเสียง ให้รวมเป็นข้อความเดียว
                pcms = []
                while True:
                    try:
                        kind, payload = self.events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "utterance":
                        pcms.append(payload)  # type: ignore[arg-type]
                    else:
                        self.events.put((kind, payload))
                        break
                if not pcms:
                    break
        except ApiError as exc:
            self.console.error(f"ถอดเสียงไม่สำเร็จ: {exc}")
            self.log.event("asr_error", error=str(exc))
            self._finish_epoch()
            return

        self.console.clear_status()
        user_text = " ".join(parts).strip()
        if self._is_echo_noise(user_text, barged):
            if barged:
                self.console.note("  (เสียงที่ตัดจังหวะเป็นเสียงสะท้อน ไม่ใช่คำพูด — "
                                  "เก็บคำตอบเดิมไว้ให้)")
                self.log.event("false_barge_in", epoch=epoch, text=user_text)
                self._repair_last_reply()
            elif user_text:
                self.log.event("noise_ignored", epoch=epoch, text=user_text)
            else:
                self.console.note("  (ไม่ได้ยินเสียงพูดชัดเจน ลองพูดอีกครั้งครับ)")
            self._finish_epoch()
            return
        self.respond(epoch, cancel, user_text)

    def respond(self, epoch: int, cancel: threading.Event, user_text: str) -> None:
        self.console.begin("🧑 คุณ", CYAN, role="user")
        self.console.write(user_text)
        self.console.end()
        self.messages.append({"role": "user", "content": user_text})
        self.log.turn("user", user_text, epoch=epoch)

        assert self.speaker is not None
        mark = len(self.speaker.finished_tags)
        self.console.status("💭", "กำลังคิด...", YELLOW)

        chunker = SentenceChunker(
            first_target=self.cfg.tts_first_chars,
            target=self.cfg.tts_chunk_chars,
            hard_max=self.cfg.tts_max_chars,
            growth=self.cfg.tts_chunk_growth,
            max_target=self.cfg.tts_max_chars,
        )
        full = ""
        seq = 0
        first_token_ms: int | None = None
        t0 = time.perf_counter()
        started = False
        error: str | None = None
        told_waiting = False

        def run_tool(name: str, args: dict) -> str:
            """โมเดลขอค้นเน็ต — บอกผู้ใช้ว่ากำลังทำอะไร จะได้ไม่เงียบหายไปเฉย ๆ"""
            nonlocal seq, told_waiting
            assert self.tools is not None
            self.console.status("🔎", describe(name, args)[:70], YELLOW)
            if self.cfg.tts_enabled and self.cfg.tts_search_filler and not told_waiting:
                told_waiting = True
                seq += 1
                self.enqueue_tts(epoch, seq, SEARCH_FILLER)
            t_tool = time.perf_counter()
            out = self.tools.run(name, args)
            self.log.event(
                "tool", epoch=epoch, name=name,
                arg=str(args.get("query") or args.get("url") or "")[:200],
                latency_ms=int((time.perf_counter() - t_tool) * 1000),
                chars=len(out),
            )
            self.console.status("💭", "กำลังเรียบเรียงคำตอบ...", YELLOW)
            return out

        use_tools = TOOLS if self.tools is not None else None
        if self.tools is not None:
            self.tools.reset()

        try:
            for delta in self.api.chat_stream(
                    self._history(), cancel, tools=use_tools,
                    run_tool=run_tool if self.tools is not None else None,
                    max_rounds=self.cfg.tool_rounds):
                if cancel.is_set():
                    break
                if not started:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                    self.console.begin("🤖 AI ", GREEN, role="assistant")
                    started = True
                full += delta
                self.console.write(delta)
                # ทยอยส่ง TTS ระหว่างที่ข้อความยังไหลอยู่ เพื่อให้เริ่มพูดเร็วที่สุด
                # (แลกกับเสียงเปลี่ยนคนทุกก้อน เพราะ OmniVoice สุ่มเสียงใหม่ทุก request
                #  พิสูจน์แล้วว่าล็อกไม่ได้ ดู README หัวข้อ "เสียงพูดของ AI")
                if not self.cfg.tts_single_request:
                    for chunk in chunker.feed(delta):
                        seq += 1
                        self.enqueue_tts(epoch, seq, chunk)
            if not cancel.is_set():
                if self.cfg.tts_single_request:
                    if full.strip() and self.cfg.tts_enabled:
                        self.console.status("🔊", "กำลังสังเคราะห์เสียง...", GREEN)
                        for piece in split_for_tts(full.strip(), self.cfg.tts_max_chars):
                            seq += 1
                            self.enqueue_tts(epoch, seq, piece)
                else:
                    rest = chunker.flush()
                    if rest:
                        seq += 1
                        self.enqueue_tts(epoch, seq, rest)
        except ApiError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)

        if error:
            if started:
                self.console.end()
            self.console.clear_status()
            self.console.error(f"เรียกโมเดลไม่สำเร็จ: {error}")
            self.log.event("chat_error", epoch=epoch, error=error)
            self._finish_epoch()
            return

        # รอให้พูดจบ (หรือถูกขัด / ถูกสั่งปิด / เกินเพดานเวลา)
        if not cancel.is_set() and self.cfg.tts_enabled:
            def tick() -> None:
                assert self.speaker is not None
                self.console.status(
                    "🔊", "AI กำลังพูด — พูดแทรกได้เลย" if self.speaker.pending()
                    else "กำลังสังเคราะห์เสียง...", GREEN)

            self.wait_for_tts(cancel, on_tick=tick)

        interrupted = cancel.is_set()
        spoken = self._spoken_since(mark, epoch)
        if started:
            self.console.end("  ⟨ถูกพูดขัด⟩" if interrupted else "")
        self.console.clear_status()

        # ไม่ได้ข้อความกลับมาเลยและไม่ได้ถูกขัด — อย่างน้อยต้องพูดอะไรสักอย่าง
        if not full.strip() and not interrupted:
            full = "ขอโทษครับ ผมยังหาคำตอบให้ไม่ได้ ลองถามใหม่อีกครั้งได้ไหมครับ"
            if not started:
                self.console.begin("🤖 AI ", GREEN, role="assistant")
            self.console.write(full)
            self.console.end()
            started = True
            seq += 1
            self.enqueue_tts(epoch, seq, full)

        used = list(self.tools.sources) if self.tools is not None else []
        if used:
            self.console.sources(used)
        if self.tools is not None and self.tools.notes:
            self.findings.append("\n".join(self.tools.notes))

        # เดิมตอนถูกขัดจะทิ้งคำตอบที่โมเดลเขียนไว้ทั้งดุ้นแล้วเก็บแค่ placeholder
        # ประวัติเลยกลายเป็น "…(ผู้ใช้พูดแทรก)" เรียงกันจนโมเดลตามเรื่องไม่ได้
        # ตอนนี้เก็บข้อความจริงเสมอ แล้วต่อท้ายด้วยหมายเหตุว่าถูกขัดตรงไหน
        answer = full.strip()
        content = answer
        if interrupted:
            content = (answer + CUT_MARK) if answer else NO_ANSWER_MARK
        if content:
            message = {"role": "assistant", "content": content}
            self.messages.append(message)
            self._last_reply = {"idx": len(self.messages) - 1,
                                "message": message, "full": answer}
        self.log.turn(
            "assistant", full.strip() or "(ไม่มีข้อความ)",
            epoch=epoch, interrupted=interrupted,
            spoken=spoken.strip() or None,
            first_token_ms=first_token_ms,
            total_ms=int((time.perf_counter() - t0) * 1000),
            sources=used or None,
        )
        self._finish_epoch()

    def _spoken_since(self, mark: int, epoch: int) -> str:
        assert self.speaker is not None
        tags = self.speaker.finished_tags[mark:]
        return "".join(t[2] for t in tags if isinstance(t, tuple) and t[0] == epoch)

    def _history(self) -> list[dict]:
        """system prompt + ผลค้นเว็บที่จำไว้ + บทสนทนาช่วงท้าย

        ผลค้นเว็บใส่เป็น role=user ที่ห่อด้วย delimiter ของเนื้อหาภายนอก
        **ห้ามเป็น system เด็ดขาด** — เนื้อหาที่ผู้โจมตีเขียนบนหน้าเว็บจะถูก
        ยกระดับเป็นคำสั่งระดับระบบทันที (P1-4) และไม่ใช้ role=tool เพราะการตัด
        ประวัติอาจตัดจนเหลือ tool ที่ไม่มี tool_calls คู่กัน แล้ว API ปฏิเสธทั้งคำขอ
        """
        keep = self.cfg.history_turns * 2
        head = self.messages[:1]
        tail = self.messages[1:][-keep:] if keep else self.messages[1:]
        if not self.findings:
            return head + tail
        note = {"role": "user",
                "content": FINDINGS_HEADER + wrap_external("\n\n".join(self.findings))}
        return head + [note] + tail

    def reset_history(self) -> None:
        self.messages = [self.cfg.system_message()]
        self.findings.clear()
        self._last_reply = None

    # ------------------------------------------------------------------- run
    def print_help(self) -> None:
        c = self.console
        c.note("  คำสั่ง: Enter = ขัดจังหวะ AI · พิมพ์ข้อความ = ส่งแบบไม่ใช้เสียง")
        c.note("          /mute ปิด-เปิดเสียง AI · /clear ล้างประวัติ · /log ที่เก็บบันทึก · /quit ออก")

    def header(self) -> None:
        c = self.console
        c.line()
        c.line(c._c(GREEN + "\033[1m", "  คุยกับ AI ด้วยเสียง แบบเรียลไทม์"))
        c.note(f"  LLM  {self.cfg.chat_model}")
        c.note(f"  ASR  {self.cfg.asr_model}")
        c.note(f"  TTS  {self.cfg.tts_model}"
               + ("" if self.cfg.tts_enabled else "  (ปิดเสียงอยู่)"))
        c.note("  เน็ต  ค้นข้อมูลปัจจุบันได้ (DuckDuckGo)" if self.cfg.web_search
               else "  เน็ต  ปิดอยู่ — ตอบจากความรู้ในโมเดลเท่านั้น")
        c.note(f"  log  {self.log.dir}")
        self.print_help()
        c.line()

    def greet(self) -> None:
        self.console.begin("🤖 AI ", GREEN, role="assistant")
        self.console.write(GREETING)
        self.console.end()
        self.log.turn("assistant", GREETING, epoch=0, greeting=True)
        self.messages.append({"role": "assistant", "content": GREETING})
        if self.cfg.tts_enabled:
            self.enqueue_tts(0, 0, GREETING)
            self.wait_for_tts(self.cancel)

    def start_workers(self) -> None:
        threading.Thread(target=self._tts_worker, name="tts", daemon=True).start()
        threading.Thread(target=self._stdin_worker, name="stdin", daemon=True).start()

    def event_loop(self) -> None:
        """วนรับเหตุการณ์จนกว่าจะสั่งออก — ใช้ร่วมกันทั้งโหมดเทอร์มินัลและโหมดเว็บ"""
        while self.running.is_set():
            if not self.busy:
                self.console.status(
                    "🎙", "พูดได้เลย..." if not self.args.no_mic
                    else "พิมพ์ข้อความแล้วกด Enter...")
            try:
                kind, payload = self.events.get(timeout=0.3)
            except queue.Empty:
                continue
            if kind == "quit":
                break
            if kind == "utterance":
                self.handle_utterance(payload)  # type: ignore[arg-type]
            elif kind == "text":
                epoch = self._new_epoch()
                self.respond(epoch, self.cancel, str(payload))

    def run(self) -> None:
        self.start_audio()
        self.header()
        self.start_workers()
        try:
            if self.args.greet:
                self.greet()      # อยู่ในนี้ด้วย ไม่งั้น Ctrl+C ตอนทักทายจะข้าม shutdown
            self.event_loop()
        except KeyboardInterrupt:
            self.console.line()
        finally:
            self.shutdown()

    def close_resources(self) -> None:
        """ปิด log และ HTTP client — เรียกซ้ำได้และห้ามโยน exception ออกไป

        แยกออกมาเพื่อให้ฝั่งเซิร์ฟเวอร์เรียกได้เองเมื่อ join เธรด session ไม่สำเร็จ
        (ถ้าเธรดค้าง `shutdown()` จะไม่มีวันถูกเรียก แล้วไฟล์บันทึก/คอนเนกชันจะรั่ว)
        """
        with self._close_lock:
            if self._resources_closed:
                return
            self._resources_closed = True
        for name, close in (("log", self.log.close), ("api", self.api.close)):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                print(f"ปิด {name} ไม่สำเร็จ: {exc!r}", file=sys.stderr)

    def shutdown(self) -> None:
        self.running.clear()
        self.interrupt("ปิดโปรแกรม", log_event=self.busy)
        if self.gate is not None:
            self.gate.stop()
        if self.mic is not None:
            self.mic.stop()
        if self.speaker is not None:
            self.audio_stuck = self.speaker.close() is False
        if self.audio_stuck:
            self.console.warn("  ปิดอุปกรณ์เสียงไม่ลง (CoreAudio ค้าง) — บังคับออกให้แล้ว")
            self.log.event("audio_close_timeout")
        self.close_resources()
        self.console.clear_status()
        self.console.line()
        self.console.note(f"  บันทึกบทสนทนาไว้ที่ {self.log.dir}")
        self.console.note(f"    · {self.log.md.name}  (อ่านง่าย)")
        self.console.note(f"    · {self.log.jsonl.name}  (เหตุการณ์ + latency)")
        self.console.line()


# ----------------------------------------------------------------- mic check
def mic_check(cfg: Config, seconds: float = 20.0) -> int:
    """มิเตอร์เสียงเข้าแบบสด — ดูว่าเสียงพูดขึ้นถึงเกณฑ์ที่ VAD ใช้จริงหรือเปล่า

    ใช้ Microphone + VoiceGate ตัวเดียวกับตอนใช้งานจริง ไม่ได้จำลองขึ้นใหม่
    """
    console = Console()
    vol = mac_input_volume()
    console.line()
    console.line("  ตรวจไมโครโฟน — พูดตามปกติจนกว่าจะครบเวลา")
    if vol is not None:
        console.note(f"  input volume ของเครื่อง: {vol}%"
                     + ("   ← ต่ำเกินไป" if vol < LOW_INPUT_VOLUME else ""))

    mic = Microphone(cfg.mic_sr, cfg.frame_ms, cfg.input_device)
    native = mic.start()
    if native != cfg.mic_sr:
        console.note(f"  ไมค์ทำงานที่ {native} Hz → แปลงเป็น {cfg.mic_sr} Hz")

    class _Silent:      # ลำโพงหลอก: ตรวจไมค์อย่างเดียว ไม่ต้องกันเสียงสะท้อน
        def recent_rms(self) -> float:
            return 0.0

    heard: list[int] = []
    live = {"level": 0.0, "threshold": cfg.vad_abs_threshold}

    def on_level(level: float, threshold: float) -> None:
        live["level"] = level
        live["threshold"] = threshold

    gate = VoiceGate(cfg, mic, _Silent(),          # type: ignore[arg-type]
                     on_speech_start=lambda: None,
                     on_utterance=lambda pcm: heard.append(pcm.size),
                     on_level=on_level)
    console.note("  กำลังวัดเสียงรบกวน อยู่เงียบ ๆ 1.5 วินาที...")
    noise = gate.calibrate(1.5)
    gain = mic.set_gain(cfg.mic_gain or min(8.0, cfg.mic_target_noise / max(noise, 1e-6)))
    gate.noise = noise * gain
    threshold = max(cfg.vad_abs_threshold, gate.noise * cfg.vad_noise_mult)
    console.note(f"  เสียงรบกวน {noise:.5f} · ขยาย {gain:.1f} เท่า → {gate.noise:.5f} · "
                 f"เกณฑ์เริ่มอัด {threshold:.4f}")
    console.line()
    gate.start()
    if mic.gain > 1.05:
        console.note("  (ถ้าเห็นค่าพุ่งชนขอบบ่อย ๆ แปลว่าขยายมากไป ลดด้วย --mic-gain)")

    width, peak, end = 46, 0.0, time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            time.sleep(0.1)
            level = float(live["level"])
            peak = max(peak, level)
            bar = int(min(1.0, level / 0.12) * width)
            hit = int(min(1.0, float(live["threshold"]) / 0.12) * width)
            row = ["·"] * width
            for i in range(bar):
                row[i] = "█"
            if hit < width:
                row[hit] = "┃"
            sys.stdout.write(f"\r  [{''.join(row)}] {level:.4f}  "
                             f"อัดแล้ว {len(heard)} ครั้ง ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        gate.stop()
        mic.stop()

    console.line()
    console.line()
    console.note(f"  ระดับสูงสุดที่วัดได้ {peak:.4f} · เกณฑ์ {threshold:.4f} · "
                 f"จับประโยคได้ {len(heard)} ครั้ง")
    if heard:
        console.line("  ผลรวม: ไมค์ใช้งานได้ ✓")
        return 0
    console.warn("  ผลรวม: ไม่ได้ยินเสียงพูดเลย ✗")
    if peak < threshold:
        console.note("  เสียงเข้ามาเบากว่าเกณฑ์ ลองอย่างใดอย่างหนึ่ง:")
        console.note("    · เพิ่ม input volume:  osascript -e 'set volume input volume 80'")
        console.note(f"    · หรือใส่ MIC_GAIN={max(2.0, threshold / max(peak, 1e-6)):.0f} ใน .env")
    return 1


# ----------------------------------------------------------------- self test
def selftest(cfg: Config) -> int:
    """ทดสอบ TTS → ASR → LLM ครบวง โดยไม่ใช้ไมโครโฟน"""
    console = Console()
    api = ApiClient(cfg)
    ok = True
    sample = "สวัสดีครับ วันนี้อากาศที่กรุงเทพเป็นอย่างไรบ้าง"
    try:
        console.note(f"[1/3] TTS ({cfg.tts_model}) ...")
        t0 = time.perf_counter()
        pcm = api.synthesize(sample, cfg.mic_sr)
        console.line(f"      ✓ ได้เสียง {pcm.size / cfg.mic_sr:.2f}s "
                     f"ใน {time.perf_counter() - t0:.2f}s")

        console.note(f"[2/3] ASR ({cfg.asr_model}) ...")
        t0 = time.perf_counter()
        text = api.transcribe(pcm, cfg.mic_sr)
        console.line(f"      ✓ ถอดได้: {text}  ({time.perf_counter() - t0:.2f}s)")

        console.note(f"[3/3] LLM ({cfg.chat_model}) สตรีม ...")
        cancel = threading.Event()
        msgs = [{"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": text or sample}]
        t0 = time.perf_counter()
        first = None
        out = ""
        for delta in api.chat_stream(msgs, cancel):
            if first is None:
                first = time.perf_counter() - t0
            out += delta
        if first is None:
            console.warn("      โมเดลไม่ส่งข้อความกลับมาเลย (อาจเป็นโมเดล reasoning "
                         "ที่ใช้ token คิดหมดก่อนตอบ ลองเพิ่ม CHAT_MAX_TOKENS)")
            ok = False
        else:
            console.line(f"      ✓ token แรก {first:.2f}s · ตอบ: {out.strip()[:120]}")
    except Exception as exc:  # noqa: BLE001
        console.error(f"ล้มเหลว: {exc}")
        ok = False
    finally:
        api.close()
    console.line()
    console.line("ผลรวม: " + ("พร้อมใช้งาน ✓" if ok else "มีปัญหา ✗"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="คุยกับ AI ด้วยเสียงแบบเรียลไทม์ (ค่าตั้งจาก .env)")
    p.add_argument("--list-devices", action="store_true", help="แสดงอุปกรณ์เสียงทั้งหมด")
    p.add_argument("--selftest", action="store_true", help="ทดสอบ API ครบวงจรโดยไม่ใช้ไมค์")
    p.add_argument("--mic-check", action="store_true",
                   help="มิเตอร์เสียงเข้าแบบสด ดูว่าไมค์ดังพอให้ VAD จับได้ไหม")
    p.add_argument("--input-device", type=int, default=None, help="หมายเลขไมโครโฟน")
    p.add_argument("--output-device", type=int, default=None, help="หมายเลขลำโพง")
    p.add_argument("--model", default=None, help="ทับค่า CHAT_MODEL ใน .env")
    p.add_argument("--headphones", action="store_true",
                   help="ใช้หูฟัง: ปิดระบบกันเสียงลำโพงย้อนเข้าไมค์ (พูดแทรกไวขึ้น)")
    p.add_argument("--no-tts", action="store_true", help="ไม่ต้องออกเสียง แสดง caption อย่างเดียว")
    p.add_argument("--no-mic", action="store_true", help="โหมดพิมพ์ ไม่ใช้ไมโครโฟน")
    p.add_argument("--no-web", action="store_true", help="ปิดการค้นข้อมูลจากอินเทอร์เน็ต")
    p.add_argument("--mic-gain", type=float, default=None,
                   help="อัตราขยายเสียงไมค์ (ไม่ใส่ = คำนวณอัตโนมัติ)")
    p.add_argument("--one-voice", action="store_true",
                   help="รอคำตอบจบแล้วค่อยพูดทีเดียว เสียงจะเป็นคนเดียวกันแต่เริ่มพูดช้าลง")
    p.add_argument("--stream-tts", action="store_true",
                   help="(ค่าเริ่มต้นอยู่แล้ว) พูดไปพร้อมกับที่ข้อความกำลังไหล")
    p.add_argument("--no-greet", dest="greet", action="store_false", help="ไม่ต้องทักทายตอนเริ่ม")
    p.add_argument("--save-audio", action="store_true", help="เก็บไฟล์เสียงที่ผู้ใช้พูดไว้ด้วย")
    p.add_argument("--threshold", type=float, default=None, help="ทับค่า VAD_ABS_THRESHOLD")
    p.set_defaults(greet=True)
    args = p.parse_args()

    if args.list_devices:
        print(list_devices())
        return 0

    cfg = load_config()
    if args.model:
        cfg.chat_model = args.model
    if args.no_tts:
        cfg.tts_enabled = False
    if args.no_web:
        cfg.web_search = False
    if args.headphones:
        cfg.echo_guard = False
    if args.save_audio:
        cfg.save_audio = True
    if args.threshold is not None:
        cfg.vad_abs_threshold = args.threshold
    if args.mic_gain is not None:
        cfg.mic_gain = args.mic_gain
    if args.stream_tts:
        cfg.tts_single_request = False
    if args.one_voice:
        cfg.tts_single_request = True
    cfg.input_device = args.input_device
    cfg.output_device = args.output_device

    if args.selftest:
        return selftest(cfg)

    if args.mic_check:
        return mic_check(cfg)

    if not cfg.asr_model and not args.no_mic:
        print("ไม่พบ ASR_MODEL ใน .env — ใช้ --no-mic เพื่อคุยแบบพิมพ์", file=sys.stderr)
        return 2

    chat = VoiceChat(cfg, args)
    chat.run()
    if chat.audio_stuck:
        # CoreAudio ค้างอยู่ ปล่อยให้ Python ปิดตัวตามปกติจะแขวนใน Py_Finalize
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
