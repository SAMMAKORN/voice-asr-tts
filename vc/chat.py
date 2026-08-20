"""แกนกลางของการคุยด้วยเสียง: เสียงเข้า → ASR → LLM (สตรีม) → TTS → เสียงออก

โมดูลนี้ไม่รู้จักทั้งบรรทัดคำสั่งและเว็บเฟรมเวิร์ก — ปลายทางเข้า/ออกถูกฉีดเข้ามา
(`Console`/`WebConsole`, `Microphone`/`WebMic`, `Speaker`/`WebSpeaker`) จึงใช้
ร่วมกันได้ทั้งโหมดเทอร์มินัล (`voice_chat.py`) และโหมดเว็บ (`web/session.py`)

เดิมคลาสนี้อยู่ในสคริปต์ CLI แล้ว `web/` import ข้ามมา ทำให้ชั้นล่างขึ้นไปพึ่งชั้นบน
และลาก PortAudio เข้ามาด้วยจน container ที่ไม่มีอุปกรณ์เสียงรันไม่ได้ (P3-21)
"""
from __future__ import annotations

import collections
import logging
import queue
import sys
import threading
import time
from typing import Callable

import numpy as np

from .api import ApiClient, ApiError, explain
from .audio import Microphone, Speaker, mac_input_volume
from .chunker import ReplyLimiter, SentenceChunker, clean_for_tts, split_for_tts
from .config import Config
from .echo import noise_reason
from .greeting import GREETINGS_MALE, GreetingWriter
from .logger import SessionLogger
from .options import RuntimeOptions
from .phase import BARGE_IN_PHASES, TurnPhase
from .thaispell import StreamSpell, ThaiSpell
from .tools import TOOLS, ToolRunner, describe, wrap_external
from .ui import Console, CYAN, GREEN, MAGENTA, RED, YELLOW
from .vad import VoiceGate

# ตัวข้อความทักทายย้ายไป vc/greeting.py แล้ว (ตอนนี้มาจาก LLM ใหม่ทุกครั้งที่เริ่มระบบ)


def search_filler_text(gender: str) -> str:
    return "ขอค้นข้อมูลสักครู่นะคะ" if gender == "female" else "ขอค้นข้อมูลสักครู่นะครับ"


def fallback_text(gender: str) -> str:
    if gender == "female":
        return "ขอโทษค่ะ ดิฉันยังหาคำตอบให้ไม่ได้ ลองถามใหม่อีกครั้งได้ไหมคะ"
    return "ขอโทษครับ ผมยังหาคำตอบให้ไม่ได้ ลองถามใหม่อีกครั้งได้ไหมครับ"


# ชื่อเดิมที่โมดูลอื่น/เทสต์เก่าอาจยัง import ตรง ๆ — ค่านี้คือค่าเริ่มต้น (เพศชาย)
GREETING = GREETINGS_MALE[0]
SEARCH_FILLER = search_filler_text("male")
CUT_MARK = " …(ผู้ใช้พูดแทรกตรงนี้ ส่วนท้ายอาจยังไม่ได้ยิน)"
NO_ANSWER_MARK = "…(ผู้ใช้พูดแทรกก่อนที่จะได้ตอบ)"
# เหตุผลเดียวที่ถือว่าเป็น "ถูกพูดขัด" จริง ๆ (มีคนพูดทับ) — เหตุผลอื่นทั้งหมด
# (กดปุ่มหยุด, กด Enter เปล่า, ปิดเสียง, ปิดโปรแกรม, ...) คือผู้ใช้สั่งหยุดเอง
BARGE_IN_REASON = "ผู้ใช้พูดแทรก"
FINDINGS_HEADER = (
    "ข้อมูลที่คุณค้นเจอไปแล้วก่อนหน้านี้ในบทสนทนาเดียวกัน "
    "ใช้ตอบต่อได้เลยโดยไม่ต้องค้นซ้ำ ถ้าผู้ใช้ถามย้ำเรื่องเดิม:\n\n")
LOW_INPUT_VOLUME = 45      # ต่ำกว่านี้บน macOS ถือว่าไมค์ถูกหรี่จนใช้งานไม่ได้
# เพดานเวลารอให้ TTS พูดจบต่อหนึ่งเทิร์น — กันลูปรอวนไม่จบถ้ามีอะไรค้างผิดปกติ
# (คำตอบยาวสุดที่เจอในบันทึกจริงใช้เวลาราว 20 วินาที จึงเผื่อไว้ 30)
TTS_WAIT_TIMEOUT = 30.0
TTS_WAIT_POLL = 0.03
# เพดานจำนวนเหตุการณ์ที่รอคิวอยู่ — client ที่ยิงเร็วกว่าที่เราประมวลผลทันต้องไม่
# ทำให้หน่วยความจำโตไม่จำกัด (P2-11) เกินเพดานจะทิ้งของเก่าสุดทิ้ง
EVENTS_MAXSIZE = 256

# log เฉพาะการเปลี่ยน phase (AC-5.5) — เปิดดูด้วย logging ระดับ DEBUG
phase_log = logging.getLogger("voicechat.phase")


class VoiceChat:
    def __init__(self, cfg: Config, args: RuntimeOptions | None = None,
                 console: Console | None = None):
        self.cfg = cfg
        # ชื่อ `args` คงไว้เพื่อไม่ให้คลาสลูก/เทสต์ที่อ้าง `self.args` พัง
        # แต่ชนิดเป็น dataclass ของแกนกลางแล้ว ไม่ใช่ Namespace ของ CLI (P3-21)
        self.args = args or RuntimeOptions()
        self.console = console or Console()   # โหมดเว็บส่ง WebConsole เข้ามาแทน
        self.api = ApiClient(cfg)
        self.log = SessionLogger(
            cfg.log_dir,
            save_audio=cfg.save_audio,
            transcript=cfg.log_transcript,
            retention_days=cfg.log_retention_days,
            tz=cfg.tz,
            meta={
                "chat_model": cfg.chat_model,
                "asr_model": cfg.asr_model,
                "tts_model": cfg.tts_model,
                "base_url": cfg.base_url,
            },
        )

        self.tools = ToolRunner(cfg) if cfg.web_search else None
        # แก้คำผิดภาษาไทยของผลถอดเสียงก่อนส่งเข้าเทิร์น (vc/thaispell.py)
        self.spell = ThaiSpell(enabled=cfg.asr_spellcheck, min_freq=cfg.spell_min_freq,
                               min_len=cfg.spell_min_len, keep=cfg.spell_keep_words)
        self.spell.prewarm()      # โหลดพจนานุกรมพื้นหลัง ไม่ให้เทิร์นแรกช้าลง
        self.greeter = GreetingWriter(cfg, self.api)   # คำทักทายใหม่ทุกครั้งที่เริ่ม
        self.messages: list[dict] = [cfg.system_message()]
        # ผลค้นเว็บของเทิร์นก่อน ๆ — เก็บแยกจาก messages เพราะต้องรอดจากการตัดประวัติ
        self.findings: collections.deque[str] = collections.deque(maxlen=cfg.keep_findings)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=EVENTS_MAXSIZE)

        self._state_lock = threading.Lock()
        self.epoch = 0
        self.cancel = threading.Event()
        self._phase = TurnPhase.IDLE    # ดู vc/phase.py — แทนธง busy ตัวเดียวของเดิม
        self.dropped_events = 0         # เหตุการณ์ที่ถูกทิ้งเพราะคิวเต็ม (P2-11)
        self.utterance_no = 0
        self._barged = False            # เทิร์นล่าสุดถูกตัดเพราะ VAD จับว่ามีคนพูด
        self._interrupt_reason = ""     # เหตุผลของการ interrupt() ครั้งล่าสุด (ดู respond())
        self._last_reply: dict | None = None   # ไว้กู้คืนถ้าที่ตัดไปเป็นเสียงหลอน
        self._last_spoken = ""          # ข้อความที่ออกลำโพงไปแล้วจริงในเทิร์นก่อน
        self.false_barge_ins = 0        # จำนวนครั้งที่ยืนยันว่าไม่ใช่การพูดแทรกจริง
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

    # ------------------------------------------------------- สถานะของเทิร์น (P2-5)
    @property
    def phase(self) -> TurnPhase:
        with self._state_lock:
            return self._phase

    @property
    def busy(self) -> bool:
        """ยังอยู่ระหว่างเทิร์นไหม — เหลือไว้เพื่อความเข้ากันได้ของ UI/คำสั่งเดิม

        **ห้ามใช้ค่านี้ตัดสินใจพูดแทรก** ต้องดู `phase` เท่านั้น เพราะ "กำลังถอดเสียง"
        กับ "กำลังพูด" ต้องได้พฤติกรรมต่างกัน (ดู vc/phase.py)
        """
        return self.phase is not TurnPhase.IDLE

    def _set_phase(self, phase: TurnPhase) -> None:
        """จุดเดียวที่เปลี่ยน phase ได้ — thread-safe และ log ทุกครั้ง"""
        with self._state_lock:
            old, self._phase = self._phase, phase
        if old is not phase:
            self._on_phase_change(old, phase)

    def _mark_speaking(self, epoch: int) -> None:
        """เสียงก้อนแรกของเทิร์นถูกส่งออกลำโพงแล้ว → GENERATING เป็น SPEAKING

        เช็ค epoch ด้วย เพราะเสียงของเทิร์นที่ถูกยกเลิกไปแล้วต้องไม่ดึงสถานะกลับ
        """
        with self._state_lock:
            if epoch != self.epoch or self._phase is not TurnPhase.GENERATING:
                return
            old, self._phase = self._phase, TurnPhase.SPEAKING
        self._on_phase_change(old, TurnPhase.SPEAKING)

    def _on_phase_change(self, old: TurnPhase, new: TurnPhase) -> None:
        """hook สำหรับคลาสลูก (โหมดเว็บส่ง phase ขึ้นไปให้ UI ด้วย)"""
        phase_log.debug("phase %s→%s (epoch=%d)", old.value, new.value, self.epoch)

    # ------------------------------------------------------------ คิวเหตุการณ์
    def put_event(self, kind: str, payload: object = None) -> bool:
        """ใส่เหตุการณ์เข้าคิวแบบมีเพดาน — คิวเต็มให้ทิ้งของเก่าสุด (P2-11)

        คืน False ถ้ามีของถูกทิ้ง เพื่อให้ผู้เรียก log ได้ว่าตามไม่ทัน
        """
        dropped = False
        while True:
            try:
                self.events.put_nowait((kind, payload))
                return not dropped
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:      # ผู้บริโภคดึงออกไปพร้อมกัน — ลองใส่ใหม่
                    continue
                dropped = True
                self.dropped_events += 1

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
        """ผู้ใช้เริ่มพูด — ตัดสินใจตาม phase ของเทิร์น ไม่ใช่ธง busy ตัวเดียว (P2-5)

        ช่วง TRANSCRIBING ห้ามขัดเด็ดขาด: เสียงที่เข้ามาตอนนั้นคือประโยคต่อของ
        เทิร์นเดียวกัน ถ้าขัดตัวเองตรงนี้ `cancel` จะถูก set ก่อนเข้า `respond()`
        แล้วเทิร์นจะหายเงียบ ๆ (ผู้ใช้ไม่ได้คำตอบและไม่มี error ให้เห็น)
        """
        if self.phase in BARGE_IN_PHASES:
            self._barged = True
            self.interrupt(BARGE_IN_REASON)

    def _on_utterance(self, pcm: np.ndarray) -> None:
        if not self.put_event("utterance", pcm):
            self.log.event("event_dropped", event="utterance",
                           total=self.dropped_events)

    def interrupt(self, reason: str, log_event: bool = True) -> None:
        self._interrupt_reason = reason
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
                spoken = clean_for_tts(text, self.cfg.tts_read_numbers)
                if self.cfg.tts_spellcheck:
                    # แก้หลัง clean_for_tts: ตัวแก้คำผิดจะได้เห็นภาษาไทยล้วน ๆ
                    # ไม่ต้องเดากับ Markdown/ลิงก์/ตัวเลขที่ยังไม่ได้แปลง
                    spoken = self.spell.fix(spoken)[0]
                if not spoken:
                    continue
                t0 = time.perf_counter()
                pcm = self.api.synthesize(spoken, self.cfg.speaker_sr)
                # ไม่เช็ค epoch ตรงนี้แล้ว — การเช็คกับการต่อคิวต้องอยู่ใน lock
                # เดียวกันของ Speaker ไม่งั้นเธรด VAD แทรก interrupt() ระหว่าง
                # สองบรรทัดได้ แล้วเสียงของเทิร์นที่ถูกยกเลิกจะเล่นต่อ (P2-12)
                if not self.speaker.play(pcm, tag=(epoch, seq, text), epoch=epoch):
                    continue
                self._mark_speaking(epoch)
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
        # ปิด session แล้วห้ามต่อคิวเพิ่ม: เธรด TTS จบไปแล้ว จะไม่มีใครมาลดตัวนับ
        # ให้อีก แล้ว `_tts_busy()` ค้างเป็น True ตลอดกาล (invariant ของ AC-2.2 พัง)
        if not self.cfg.tts_enabled or not self.running.is_set():
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
                self.put_event("quit")
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
            self.put_event("text", text)

    # ------------------------------------------------------------------- turn
    def _new_epoch(self, phase: TurnPhase = TurnPhase.TRANSCRIBING) -> int:
        """เปิดเทิร์นใหม่: เลื่อน epoch, ล้าง cancel, ตั้ง phase เริ่มต้นของเทิร์น

        ทั้งสามอย่างต้องเกิดพร้อมกันใน lock เดียว ไม่งั้นเธรด VAD อาจเห็น epoch ใหม่
        แต่ phase เก่า (หรือกลับกัน) แล้วตัดสินใจพูดแทรกผิดจังหวะ
        """
        with self._state_lock:
            self.epoch += 1
            self.cancel = threading.Event()
            old, self._phase = self._phase, phase
            epoch = self.epoch
        # เปิดประตูลำโพงให้เฉพาะเสียงของเทิร์นนี้ (fence ของ P2-12 อยู่ใน Speaker)
        if self.speaker is not None:
            self.speaker.set_epoch(epoch)
        if old is not phase:
            self._on_phase_change(old, phase)
        return epoch

    def _finish_epoch(self) -> None:
        self._set_phase(TurnPhase.IDLE)

    def _noise_reason(self, text: str, barged: bool) -> str | None:
        """เสียงที่เพิ่งจับได้เป็นคนพูดจริงไหม — คืนเหตุผลถ้าไม่ใช่ (ดู vc/echo.py)

        เทียบกับ `_last_spoken` ซึ่งเป็นข้อความที่ออกลำโพงไปแล้วจริง ไม่ใช่คำตอบเต็ม
        เพราะเสียงที่วนกลับเข้าไมค์ได้มีแค่ส่วนที่ถูกเล่นออกไปแล้วเท่านั้น
        """
        return noise_reason(text, self._last_spoken, lang_hint=self.cfg.lang_hint,
                            min_chars=self.cfg.barge_in_min_chars, barged=barged)

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
        """หนึ่งเทิร์นเต็ม: ถอดเสียง → ตอบ — ต้องคืนสถานะเป็น IDLE ทุกเส้นทาง

        ทั้งก้อนอยู่ใน try/finally ที่เรียก `_finish_epoch()` เสมอ (AC-6.6) และดัก
        `Exception` กว้างระดับเทิร์น: เน็ตกระตุกหรือบั๊กในเทิร์นเดียวต้องไม่ล้ม
        session ทั้งอัน (บั๊กเดิม: `httpx.ConnectError` ทะลุถึง `run()` แล้วปิดเลย)
        """
        epoch = self._new_epoch(TurnPhase.TRANSCRIBING)
        cancel = self.cancel
        barged, self._barged = self._barged, False
        try:
            user_text = self._transcribe_turn(epoch, first)
            if user_text is None:
                return
            reason = self._noise_reason(user_text, barged)
            if reason:
                if barged:
                    self.false_barge_ins += 1
                    self.console.note("  (เสียงที่ตัดจังหวะเป็นเสียงสะท้อน ไม่ใช่คำพูด — "
                                      "เก็บคำตอบเดิมไว้ให้)")
                    self.log.event("false_barge_in", epoch=epoch, text=user_text,
                                   reason=reason, total=self.false_barge_ins)
                    self._repair_last_reply()
                elif user_text:
                    self.log.event("noise_ignored", epoch=epoch, text=user_text,
                                   reason=reason)
                else:
                    self.console.note("  (ไม่ได้ยินเสียงพูดชัดเจน ลองพูดอีกครั้งครับ)")
                return
            self.respond(epoch, cancel, user_text)
        except Exception as exc:  # noqa: BLE001 — กันเทิร์นเดียวล้มทั้ง session
            self.console.clear_status()
            self.console.error(f"เทิร์นนี้ไม่สำเร็จ: {exc}")
            self.log.event("turn_error", epoch=epoch, error=repr(exc))
        finally:
            self._finish_epoch()

    def _transcribe_turn(self, epoch: int, first: np.ndarray) -> str | None:
        """ถอดเสียงทุกประโยคของเทิร์นนี้ — คืน None ถ้าถอดไม่สำเร็จ

        ระหว่างถอดเสียงผู้ใช้ยังพูดต่อได้ (phase = TRANSCRIBING จึงไม่นับเป็นการ
        พูดแทรก) ประโยคที่เข้ามาเพิ่มจะถูกดึงจากคิวมาต่อเป็นข้อความเดียวกัน
        """
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
                pcms = self._pending_utterances()
                if not pcms:
                    break
        except ApiError as exc:
            self.console.clear_status()
            self.console.error(f"ถอดเสียงไม่สำเร็จ: {exc}")
            self.log.event("asr_error", epoch=epoch, error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 — ชนิดอื่นที่หลุดมาจากชั้นล่าง
            self.console.clear_status()
            self.console.error(f"ถอดเสียงไม่สำเร็จ: {exc}")
            self.log.event("asr_error", epoch=epoch, error=repr(exc))
            return None

        self.console.clear_status()
        return self._spellcheck(epoch, " ".join(parts).strip())

    def _spellcheck(self, epoch: int, text: str) -> str:
        """แก้คำผิดภาษาไทยของผลถอดเสียง — ด่านเดียวก่อนข้อความเข้าสู่เทิร์น

        แก้ที่นี่ที่เดียว ทุกอย่างปลายน้ำ (การเทียบเสียงสะท้อน, ประวัติสนทนาที่ส่งให้
        โมเดล, หน้าจอ, transcript) จึงเห็นข้อความเดียวกันหมด · ข้อความดิบก่อนแก้ยัง
        อยู่ครบใน event `asr` ของ session.jsonl ถ้าต้องย้อนดูว่า ASR ได้ยินว่าอะไร
        """
        if not text:
            return text
        fixed, changes = self.spell.fix(text)
        if changes:
            self.log.event("spell_fix", epoch=epoch, side="asr",
                           count=len(changes),
                           detail=", ".join(f"{a}→{b}" for a, b in changes))
        return fixed

    def _pending_utterances(self) -> list[np.ndarray]:
        """ดึงเฉพาะเสียงที่รอในคิวออกมา (เหตุการณ์ชนิดอื่นคืนกลับเข้าคิวตามลำดับ)"""
        pcms: list[np.ndarray] = []
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return pcms
            if kind == "utterance":
                pcms.append(payload)  # type: ignore[arg-type]
            else:
                self.put_event(kind, payload)
                return pcms

    def respond(self, epoch: int, cancel: threading.Event, user_text: str) -> None:
        """ตอบหนึ่งเทิร์น — การันตีว่าสถานะกลับเป็น IDLE ทุกเส้นทางที่ออกจากที่นี่"""
        self._set_phase(TurnPhase.GENERATING)
        try:
            self._respond(epoch, cancel, user_text)
        finally:
            self._finish_epoch()

    def _respond(self, epoch: int, cancel: threading.Event, user_text: str) -> None:
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
        at = self.cfg.now()      # เวลาของเทิร์นนี้ ใช้ร่วมกันทั้ง prompt และตัวซ่อมวลี
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
                self.enqueue_tts(epoch, seq, search_filler_text(self.cfg.voice_gender))
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

        # เพดานความยาวคำตอบ บังคับในโค้ด ไม่พึ่ง system prompt อย่างเดียว (P2-15)
        limiter = ReplyLimiter(self.cfg.reply_max_sentences, self.cfg.reply_max_chars)
        stream = self.api.chat_stream(
            self._history(at), cancel, tools=use_tools,
            run_tool=run_tool if self.tools is not None else None,
            max_rounds=self.cfg.tool_rounds)

        # แก้คำผิดของคำตอบตั้งแต่ก่อนขึ้นจอ — จอ ประวัติ transcript และเสียงจึงตรงกัน
        # `supplied_phrases(at)` ต้องมาจาก `at` ตัวเดียวกับที่ประกอบ system prompt
        # ไม่งั้นพอข้ามนาที ตัวซ่อมจะดึงเวลาที่โมเดลเขียนถูกให้ย้อนไปนาทีก่อนหน้า
        speller = StreamSpell(self.spell if self.cfg.chat_spellcheck
                              else ThaiSpell(enabled=False),
                              self.cfg.supplied_phrases(at))

        def emit(text: str) -> None:
            """ส่งข้อความที่ผ่านเพดานแล้วออกทั้งจอและคิวเสียง"""
            nonlocal full, seq, started
            if not text:
                return
            if not started:
                self.console.begin("🤖 AI ", GREEN, role="assistant")
                started = True
            full += text
            self.console.write(text)
            # ทยอยส่ง TTS ระหว่างที่ข้อความยังไหลอยู่ เพื่อให้เริ่มพูดเร็วที่สุด
            # (แลกกับเสียงเปลี่ยนคนทุกก้อน เพราะ OmniVoice สุ่มเสียงใหม่ทุก request
            #  พิสูจน์แล้วว่าล็อกไม่ได้ ดู README หัวข้อ "เสียงพูดของ AI")
            if not self.cfg.tts_single_request:
                for chunk in chunker.feed(text):
                    seq += 1
                    self.enqueue_tts(epoch, seq, chunk)

        def push(text: str) -> None:
            """ทางเข้าเดียวของข้อความจากโมเดล — ผ่านตัวแก้คำก่อนถึง `emit()`

            จับเวลา "token แรก" ที่นี่ ไม่ใช่ใน `emit()` เพราะตัวแก้คำกันข้อความ
            ท้ายไว้รอคำที่ยังพิมพ์ไม่จบ ถ้าไปจับตอนปล่อยผ่าน สถิติจะกลายเป็นเวลา
            ของตัวแก้คำแทนที่จะเป็นเวลาที่โมเดลเริ่มตอบจริง
            """
            nonlocal first_token_ms
            if text and first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            emit(speller.feed(text))

        try:
            for delta in stream:
                if cancel.is_set():
                    break
                passed, capped = limiter.feed(delta)
                push(passed)
                if capped:
                    # ปิดสตรีมให้สะอาด (ไม่ทิ้ง connection ค้าง ไม่มี chunk หลุดต่อ)
                    stream.close()
                    self.console.status("✂️", "ตอบครบตามความยาวที่กำหนดแล้ว", YELLOW)
                    self.log.event("reply_capped", epoch=epoch, chars=len(full),
                                   sentences=limiter.sentences,
                                   max_chars=self.cfg.reply_max_chars,
                                   max_sentences=self.cfg.reply_max_sentences)
                    break
            else:
                push(limiter.flush())      # สตรีมจบเอง — ปล่อยส่วนที่ค้างในเพดานออก
            # ถูกพูดแทรก = ทิ้งของที่ค้างในตัวแก้คำ ผู้ใช้ไม่ได้ยินท่อนนั้นอยู่แล้ว
            if not cancel.is_set():
                emit(speller.flush())
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
        finally:
            stream.close()          # ปิดสตรีมเสมอ แม้ออกทางข้อผิดพลาด

        if error:
            if started:
                self.console.end()
            self.console.clear_status()
            self.console.error(f"เรียกโมเดลไม่สำเร็จ: {error}")
            self.log.event("chat_error", epoch=epoch, error=error)
            self._show_failure(explain(error))
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
        # เก็บ "สิ่งที่ผู้ใช้ได้ยินจริง" ไว้เทียบกับเสียงที่อาจสะท้อนกลับเข้าไมค์ (P3-23)
        self._last_spoken = spoken
        if started:
            # "ถูกพูดขัด" เฉพาะตอนมีคนพูดทับจริง ๆ — ถูกสั่งหยุด (ปุ่มหยุด/กด Enter/
            # ปิดเสียง/ปิดโปรแกรม ฯลฯ) ไม่ใช่การพูดขัด ป้ายต้องบอกว่า "ถูกหยุดพูด"
            label = ("ถูกพูดขัด" if self._interrupt_reason == BARGE_IN_REASON
                     else "ถูกหยุดพูด")
            self.console.end(f"  ⟨{label}⟩" if interrupted else "")
        self.console.clear_status()

        # ไม่ได้ข้อความกลับมาเลยและไม่ได้ถูกขัด — อย่างน้อยต้องพูดอะไรสักอย่าง
        if not full.strip() and not interrupted:
            full = fallback_text(self.cfg.voice_gender)
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
        if speller.changes:
            self.log.event("spell_fix", epoch=epoch, side="reply",
                           count=len(speller.changes),
                           detail=", ".join(f"{a}→{b}"
                                            for a, b in speller.changes)[:500])
        self.log.turn(
            "assistant", full.strip() or "(ไม่มีข้อความ)",
            epoch=epoch, interrupted=interrupted,
            spoken=spoken.strip() or None,
            first_token_ms=first_token_ms,
            total_ms=int((time.perf_counter() - t0) * 1000),
            sources=used or None,
        )

    def _spoken_since(self, mark: int, epoch: int) -> str:
        assert self.speaker is not None
        tags = self.speaker.finished_tags[mark:]
        return "".join(t[2] for t in tags if isinstance(t, tuple) and t[0] == epoch)

    def _show_failure(self, text: str) -> None:
        """แจ้งความล้มเหลวในช่องบทสนทนาเอง ไม่ใช่แค่ในแผงบันทึก

        `console.error()` ลงแต่แผงบันทึกระบบ ผู้ใช้จึงเห็นคำถามของตัวเองแล้วเงียบ
        แยกไม่ออกว่าระบบพังหรือ AI เลือกจะไม่ตอบ (เจอจริงตอน CHAT_MODEL ใน .env
        ชี้ไปโมเดลที่ไม่มีสิทธิ์เรียก: ทุกเทิร์นได้ 403 แต่หน้าจอว่างเปล่าสนิท)

        ใช้ role=system ไม่ใช่ assistant — ข้อความนี้ไม่ได้ถูกพูดออกเสียง ไม่ถูก
        เก็บลง `self.messages` และต้องหน้าตาต่างจากคำตอบจริงของ AI
        """
        self.console.begin("⚠️ ระบบ", RED, role="system")
        self.console.write(text)
        self.console.end()

    def _history(self, at: object = None) -> list[dict]:
        """system prompt + ผลค้นเว็บที่จำไว้ + บทสนทนาช่วงท้าย

        ผลค้นเว็บใส่เป็น role=user ที่ห่อด้วย delimiter ของเนื้อหาภายนอก
        **ห้ามเป็น system เด็ดขาด** — เนื้อหาที่ผู้โจมตีเขียนบนหน้าเว็บจะถูก
        ยกระดับเป็นคำสั่งระดับระบบทันที (P1-4) และไม่ใช้ role=tool เพราะการตัด
        ประวัติอาจตัดจนเหลือ tool ที่ไม่มี tool_calls คู่กัน แล้ว API ปฏิเสธทั้งคำขอ
        """
        keep = self.cfg.history_turns * 2
        # ประกอบ system prompt ใหม่ทุกเทิร์น ไม่ใช้ตัวที่เก็บไว้ตอนเปิด session:
        # ในนั้นมีวันเวลาปัจจุบันฝังอยู่ (P3-20) เซสชันที่เปิดค้างข้ามคืนจะบอกโมเดล
        # ว่าตอนนี้คือเวลาที่เปิดหน้าเว็บ แล้วคำตอบเรื่อง "ตอนนี้กี่โมง" ผิดทั้งวัน
        head = [self.cfg.system_message(at)]
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
        c.note(f"  TTS  {self.cfg.tts_label}"
               + ("" if self.cfg.tts_enabled else "  (ปิดเสียงอยู่)"))
        c.note("  เน็ต  ค้นข้อมูลปัจจุบันได้ (DuckDuckGo)" if self.cfg.web_search
               else "  เน็ต  ปิดอยู่ — ตอบจากความรู้ในโมเดลเท่านั้น")
        c.note(f"  log  {self.log.dir}")
        self.print_help()
        c.line()

    def greet(self) -> None:
        """ทักทายตอนเริ่ม — ต้องอยู่ใต้ epoch จริงและ phase = SPEAKING (P2-5)

        เดิมคำทักทายทำงานนอก `_new_epoch()` จึงพูดแทรกไม่ได้เลย (phase = IDLE →
        `_on_speech_start()` ไม่สั่งหยุดอะไร) ขัดกับที่โฆษณาไว้ทั้งใน README
        และในตัวข้อความทักทายเอง

        ตัวข้อความมาจาก LLM ใหม่ทุกครั้ง (`vc/greeting.py`) การยิงจึงอยู่ใน epoch นี้
        ด้วยและรับ `cancel` ตัวเดียวกัน — ผู้ใช้ที่พูดใส่ทันทีตอนเปิดระบบต้องขัดได้
        ตั้งแต่ตอนที่คำทักทายยังแต่งไม่เสร็จ ไม่ใช่ต้องรอให้พูดจบก่อน
        """
        epoch = self._new_epoch(TurnPhase.SPEAKING)
        cancel = self.cancel
        mark = len(self.speaker.finished_tags) if self.speaker is not None else 0
        try:
            self.console.status("💭", "กำลังคิดคำทักทาย...", YELLOW)
            greeting = self.greeter.make(cancel)
            self.console.clear_status()
            if not self.running.is_set() or cancel.is_set():
                return      # ปิดแท็บ/พูดแทรกระหว่างที่ยังแต่งคำทักทายไม่เสร็จ
            self.console.begin("🤖 AI ", GREEN, role="assistant")
            self.console.write(greeting)
            self.console.end()
            self.log.turn("assistant", greeting, epoch=epoch, greeting=True)
            self.messages.append({"role": "assistant", "content": greeting})
            if self.cfg.tts_enabled:
                self.enqueue_tts(epoch, 0, greeting)
                self.wait_for_tts(cancel)
                # คำทักทายก็สะท้อนเข้าไมค์ได้เหมือนคำตอบปกติ (P3-23)
                self._last_spoken = self._spoken_since(mark, epoch)
        finally:
            self._finish_epoch()

    def start_workers(self) -> None:
        threading.Thread(target=self._tts_worker, name="tts", daemon=True).start()
        threading.Thread(target=self._stdin_worker, name="stdin", daemon=True).start()

    def event_loop(self) -> None:
        """วนรับเหตุการณ์จนกว่าจะสั่งออก — ใช้ร่วมกันทั้งโหมดเทอร์มินัลและโหมดเว็บ"""
        while self.running.is_set():
            if self.phase is TurnPhase.IDLE:
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
                # ข้อความพิมพ์ไม่ต้องถอดเสียง — เข้าสู่ช่วงคิดคำตอบได้เลย
                epoch = self._new_epoch(TurnPhase.GENERATING)
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

    def log_input_stats(self) -> None:
        """สรุปตัวเลขคุณภาพเสียงเข้าตอนจบ session (P3-23)

        เดิมนับ `overflows`/`clipped`/`dropped` ไว้แต่ไม่เคยแสดงที่ไหนเลย ทั้งที่เป็น
        คำตอบตรง ๆ ของ "ทำไมมันไม่ได้ยินที่พูด" — overflow/dropped = เฟรมหาย,
        clipped = ขยายเสียงมากไปจนเสียงเพี้ยน
        """
        mic = self.mic
        stats = {
            "overflows": int(getattr(mic, "overflows", 0) or 0),
            "clipped": int(getattr(mic, "clipped", 0) or 0),
            "dropped_frames": int(getattr(mic, "dropped", 0) or 0),
            "dropped_events": self.dropped_events,
            "false_barge_ins": self.false_barge_ins,
        }
        self.log.event("input_stats", **stats)
        if any(stats.values()):
            self.console.note(
                f"  คุณภาพเสียงเข้า: เฟรมล้น {stats['overflows']} · "
                f"เฟรมถูกทิ้ง {stats['dropped_frames']} · "
                f"เสียงคลิป {stats['clipped']} · "
                f"เหตุการณ์ถูกทิ้ง {stats['dropped_events']} · "
                f"พูดแทรกที่เป็นเสียงสะท้อน {stats['false_barge_ins']} ครั้ง")

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
        self.log_input_stats()      # ต้องอยู่ก่อน close_resources() ไม่งั้นเขียนไม่ลง
        self.close_resources()
        self.console.clear_status()
        self.console.line()
        self.console.note(f"  บันทึกบทสนทนาไว้ที่ {self.log.dir}")
        if self.log.transcript:
            self.console.note(f"    · {self.log.md.name}  (อ่านง่าย)")
        else:
            self.console.note("    · ปิดการเก็บคำพูดไว้ (LOG_TRANSCRIPT=0)")
        self.console.note(f"    · {self.log.jsonl.name}  (เหตุการณ์ + latency)")
        self.console.line()

