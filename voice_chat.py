#!/usr/bin/env python3
"""คุยโต้ตอบกับ AI ด้วยเสียงแบบเรียลไทม์ (ไทย)

  เสียงเข้า → ASR → LLM (สตรีม) → TTS → เสียงออก
  พูดแทรกได้ตลอดเวลา · แสดง caption ทันที · บันทึกบทสนทนาลง logs/

ใช้ค่า API/โมเดลจาก .env ทั้งหมด
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

from vc.api import ApiClient, ApiError
from vc.audio import Microphone, Speaker, list_devices
from vc.chunker import SentenceChunker, clean_for_tts
from vc.config import Config, load_config
from vc.logger import SessionLogger
from vc.tools import TOOLS, ToolRunner, describe
from vc.ui import Console, CYAN, GREEN, GRAY, MAGENTA, YELLOW
from vc.vad import VoiceGate

GREETING = "สวัสดีครับ ผมพร้อมคุยแล้ว พูดได้เลยครับ พูดแทรกได้ตลอดเวลา"
SEARCH_FILLER = "ขอค้นข้อมูลสักครู่นะครับ"


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
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._state_lock = threading.Lock()
        self.epoch = 0
        self.cancel = threading.Event()
        self.busy = False               # กำลังคิด/พูด → พูดแทรกได้
        self.utterance_no = 0

        self.tts_queue: queue.Queue[tuple[int, int, str] | None] = queue.Queue()
        self._inflight = 0
        self._inflight_lock = threading.Lock()

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
        self.console.clear_status()
        self.console.note(f"  ระดับเสียงรบกวน {noise:.4f} · "
                          f"เกณฑ์เริ่มอัด {max(self.cfg.vad_abs_threshold, noise * self.cfg.vad_noise_mult):.4f}")
        self.log.event("calibrated", noise=round(noise, 5))
        self.gate.start()

    # -------------------------------------------------------- callback จาก VAD
    def _on_speech_start(self) -> None:
        """ผู้ใช้เริ่มพูด — ถ้า AI กำลังคิดหรือพูดอยู่ ให้หยุดทันที"""
        with self._state_lock:
            busy = self.busy
        if busy:
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
                self.messages = [self.cfg.system_message()]
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

    def handle_utterance(self, first: np.ndarray) -> None:
        epoch = self._new_epoch()
        cancel = self.cancel
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
        if not user_text:
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

        chunker = SentenceChunker()
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
            if self.cfg.tts_enabled and not told_waiting:
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
                for chunk in chunker.feed(delta):
                    seq += 1
                    self.enqueue_tts(epoch, seq, chunk)
            if not cancel.is_set():
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

        # รอให้พูดจบ (หรือถูกขัด)
        if not cancel.is_set() and self.cfg.tts_enabled:
            self.console.status("🔊", "AI กำลังพูด — พูดแทรกได้เลย", GREEN)
            while not cancel.is_set() and (self._tts_busy() or self.speaker.pending()):
                time.sleep(0.03)

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

        content = full.strip()
        if interrupted:
            heard = spoken.strip()
            content = (f"{heard} …(ผู้ใช้พูดแทรกตรงนี้)" if heard
                       else "…(ผู้ใช้พูดแทรกก่อนที่จะได้ตอบ)")
        if content:
            self.messages.append({"role": "assistant", "content": content})
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
        keep = self.cfg.history_turns * 2
        head = self.messages[:1]
        tail = self.messages[1:][-keep:] if keep else self.messages[1:]
        return head + tail

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
            while self._tts_busy() or (self.speaker and self.speaker.pending()):
                time.sleep(0.05)

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

        if self.args.greet:
            self.greet()

        try:
            self.event_loop()
        except KeyboardInterrupt:
            self.console.line()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running.clear()
        self.interrupt("ปิดโปรแกรม", log_event=self.busy)
        if self.gate is not None:
            self.gate.stop()
        if self.mic is not None:
            self.mic.stop()
        if self.speaker is not None:
            self.speaker.close()
        self.log.close()
        self.api.close()
        self.console.clear_status()
        self.console.line()
        self.console.note(f"  บันทึกบทสนทนาไว้ที่ {self.log.dir}")
        self.console.note(f"    · {self.log.md.name}  (อ่านง่าย)")
        self.console.note(f"    · {self.log.jsonl.name}  (เหตุการณ์ + latency)")
        self.console.line()


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
    p.add_argument("--input-device", type=int, default=None, help="หมายเลขไมโครโฟน")
    p.add_argument("--output-device", type=int, default=None, help="หมายเลขลำโพง")
    p.add_argument("--model", default=None, help="ทับค่า CHAT_MODEL ใน .env")
    p.add_argument("--headphones", action="store_true",
                   help="ใช้หูฟัง: ปิดระบบกันเสียงลำโพงย้อนเข้าไมค์ (พูดแทรกไวขึ้น)")
    p.add_argument("--no-tts", action="store_true", help="ไม่ต้องออกเสียง แสดง caption อย่างเดียว")
    p.add_argument("--no-mic", action="store_true", help="โหมดพิมพ์ ไม่ใช้ไมโครโฟน")
    p.add_argument("--no-web", action="store_true", help="ปิดการค้นข้อมูลจากอินเทอร์เน็ต")
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
    cfg.input_device = args.input_device
    cfg.output_device = args.output_device

    if args.selftest:
        return selftest(cfg)

    if not cfg.asr_model and not args.no_mic:
        print("ไม่พบ ASR_MODEL ใน .env — ใช้ --no-mic เพื่อคุยแบบพิมพ์", file=sys.stderr)
        return 2

    VoiceChat(cfg, args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
