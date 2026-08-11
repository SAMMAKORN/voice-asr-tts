"""หนึ่ง WebSocket = หนึ่ง session คุยด้วยเสียง

ใช้ VoiceChat ตัวเดิมทั้งดุ้น (epoch, คิว TTS, การพูดแทรก, การบันทึก log)
เปลี่ยนแค่ปลายทางเข้า-ออก: ไมค์/ลำโพง/caption ย้ายไปอยู่บนเบราว์เซอร์
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from vc.api import ApiClient, ApiError
from vc.config import Config
from vc.vad import VoiceGate
from voice_chat import GREETING, VoiceChat

from .bridge import Outbox, WebConsole, WebMic, WebSpeaker

LEVEL_INTERVAL = 0.1      # ส่งระดับเสียงให้ UI วาดมิเตอร์ ~10 ครั้ง/วินาที

# เหตุการณ์ที่ส่งต่อให้หน้าเว็บโชว์เป็นตัวเลข latency (ตัวเนื้อความไม่ต้องส่งซ้ำ)
METRIC_EVENTS = {"asr", "tts", "message", "interrupt", "tool",
                 "asr_error", "tts_error", "chat_error"}
METRIC_SKIP = {"text", "spoken", "audio_file", "meta", "sources"}


class WebSession(VoiceChat):
    def __init__(self, cfg: Config, out: Outbox):
        args = argparse.Namespace(
            no_mic=False, greet=True, save_audio=cfg.save_audio,
            headphones=not cfg.echo_guard, no_tts=not cfg.tts_enabled,
        )
        super().__init__(cfg, args, console=WebConsole(out))
        self.out = out
        self.mic = WebMic(cfg.mic_sr, cfg.frame_ms)
        self.speaker = WebSpeaker(out, cfg.speaker_sr)
        self._level_at = 0.0
        self._closed = threading.Event()
        self._forward_metrics()

    def _forward_metrics(self) -> None:
        """ดัก log.event ให้ส่ง latency ขึ้นไปโชว์บนหน้าเว็บด้วย (ยังเขียนไฟล์เหมือนเดิม)"""
        inner = self.log.event

        def event(kind: str, **fields) -> None:
            inner(kind, **fields)
            if kind not in METRIC_EVENTS:
                return
            payload = {k: v for k, v in fields.items()
                       if k not in METRIC_SKIP
                       and isinstance(v, (int, float, str, bool, type(None)))}
            self.out.json("metric", kind=kind, **payload)

        self.log.event = event  # type: ignore[method-assign]

    # -------------------------------------------------------------- ตั้งต้นเสียง
    def start_audio(self) -> None:
        assert self.mic is not None and self.speaker is not None
        self.gate = VoiceGate(
            self.cfg, self.mic, self.speaker,
            on_speech_start=self._on_speech_start,
            on_utterance=self._on_utterance,
            on_level=self._on_level,
        )
        self.console.status("🎚", "กำลังวัดเสียงรบกวนรอบข้าง อยู่เงียบ ๆ ครู่หนึ่ง...")
        # รอจนเบราว์เซอร์ได้สิทธิ์ไมค์แล้วเริ่มสตรีมเข้ามาจริง ๆ
        if not self.mic.started.wait(timeout=30.0):
            raise RuntimeError("ไม่ได้รับเสียงจากเบราว์เซอร์ (ยังไม่ได้อนุญาตไมโครโฟน?)")
        noise = self.gate.calibrate(1.2)
        self.console.clear_status()
        threshold = max(self.cfg.vad_abs_threshold, noise * self.cfg.vad_noise_mult)
        self.out.json("calibrated", noise=round(noise, 5), threshold=round(threshold, 5))
        self.log.event("calibrated", noise=round(noise, 5))
        self.gate.start()

    def start_workers(self) -> None:
        """เหมือนเดิมแต่ไม่มีเธรดอ่าน stdin — ฝั่งเว็บรับคำสั่งผ่าน WebSocket"""
        threading.Thread(target=self._tts_worker, name="tts", daemon=True).start()

    # ---------------------------------------------------------------- callbacks
    def _on_speech_start(self) -> None:
        self.out.json("speech", state="start")
        super()._on_speech_start()

    def _on_utterance(self, pcm: np.ndarray) -> None:
        self.out.json("speech", state="end",
                      ms=int(pcm.size / self.cfg.mic_sr * 1000))
        super()._on_utterance(pcm)

    def _on_level(self, level: float, threshold: float) -> None:
        now = time.monotonic()
        if now - self._level_at < LEVEL_INTERVAL:
            return
        self._level_at = now
        self.out.json("level", rms=round(level, 4), threshold=round(threshold, 4))

    # ------------------------------------------------------- ข้อความจากเบราว์เซอร์
    def feed_audio(self, data: bytes) -> None:
        assert self.mic is not None
        if len(data) >= 2:
            self.mic.feed(np.frombuffer(data, dtype="<i2"))

    def handle_client(self, msg: dict) -> None:
        assert self.speaker is not None
        kind = msg.get("type")
        if kind == "played":
            self.speaker.note_played(int(msg.get("epoch", 0)), int(msg.get("seq", 0)))
        elif kind == "level":
            self.speaker.note_level(float(msg.get("out", 0.0)))
        elif kind == "text":
            text = str(msg.get("text", "")).strip()
            if text:
                self.events.put(("text", text))
        elif kind == "interrupt":
            if self.busy:
                self.interrupt("ผู้ใช้กดหยุด")
        elif kind == "mute":
            self.cfg.tts_enabled = not bool(msg.get("on"))
            if not self.cfg.tts_enabled:
                self.interrupt("ปิดเสียง", log_event=False)
            self.out.json("setting", key="mute", on=not self.cfg.tts_enabled)
        elif kind == "echo_guard":
            self.cfg.echo_guard = bool(msg.get("on"))
            self.out.json("setting", key="echo_guard", on=self.cfg.echo_guard)
        elif kind == "clear":
            self.messages = [self.cfg.system_message()]
            self.log.event("history_cleared")
            self.out.json("cleared")
        elif kind == "quit":
            self.request_stop()

    def request_stop(self) -> None:
        """สั่งปิด session จากฝั่ง asyncio (client ตัดการเชื่อมต่อ/กดออก)"""
        self._closed.set()
        self.running.clear()
        if self.mic is not None:
            self.mic.started.set()       # ปลดล็อกกรณีค้างรออยู่ตอน calibrate
        self.events.put(("quit", None))

    # --------------------------------------------------------------------- run
    def header(self) -> None:
        self.out.json(
            "ready",
            chat_model=self.cfg.chat_model,
            asr_model=self.cfg.asr_model,
            tts_model=self.cfg.tts_model,
            base_url=self.cfg.base_url,
            mic_sr=self.cfg.mic_sr,
            speaker_sr=self.cfg.speaker_sr,
            frame_ms=self.cfg.frame_ms,
            tts_enabled=self.cfg.tts_enabled,
            echo_guard=self.cfg.echo_guard,
            web_search=self.cfg.web_search,
            log_dir=str(self.log.dir),
        )

    def print_help(self) -> None:
        pass

    def greet(self) -> None:
        if self._closed.is_set():
            return
        super().greet()

    def run(self) -> None:
        try:
            self.start_audio()
            self.header()
            self.start_workers()
            if self.args.greet:
                self.greet()
            self.event_loop()
        except Exception as exc:  # noqa: BLE001
            self.console.error(str(exc))
            self.log.event("session_error", error=repr(exc))
        finally:
            try:
                self.shutdown()
            finally:
                self.out.json("closed")
                self.out.close()


# --------------------------------------------------------------- ตรวจระบบ (onboarding)
def run_selftest(cfg: Config) -> dict:
    """เรียก TTS → ASR → LLM ครบวง แล้วคืนผลเป็นโครงสร้างให้หน้าเว็บแสดงทีละขั้น"""
    api = ApiClient(cfg)
    sample = "สวัสดีครับ วันนี้อากาศที่กรุงเทพเป็นอย่างไรบ้าง"
    steps: list[dict] = []

    def add(name: str, model: str, ok: bool, detail: str, ms: int) -> None:
        steps.append({"name": name, "model": model, "ok": ok,
                      "detail": detail, "ms": ms})

    try:
        t0 = time.perf_counter()
        pcm = api.synthesize(sample, cfg.mic_sr)
        add("TTS", cfg.tts_model, True,
            f"สังเคราะห์เสียงได้ {pcm.size / cfg.mic_sr:.2f} วินาที",
            int((time.perf_counter() - t0) * 1000))

        t0 = time.perf_counter()
        text = api.transcribe(pcm, cfg.mic_sr)
        add("ASR", cfg.asr_model, bool(text), text or "ถอดเสียงไม่ได้ข้อความ",
            int((time.perf_counter() - t0) * 1000))

        t0 = time.perf_counter()
        cancel = threading.Event()
        msgs = [{"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": text or sample}]
        out, first = "", None
        for delta in api.chat_stream(msgs, cancel):
            if first is None:
                first = int((time.perf_counter() - t0) * 1000)
            out += delta
            if len(out) > 400:
                cancel.set()
        add("LLM", cfg.chat_model, first is not None,
            out.strip()[:160] or "โมเดลไม่ส่งข้อความกลับมา",
            first or int((time.perf_counter() - t0) * 1000))
    except ApiError as exc:
        steps.append({"name": "ผิดพลาด", "model": "", "ok": False,
                      "detail": str(exc)[:300], "ms": 0})
    except Exception as exc:  # noqa: BLE001
        steps.append({"name": "ผิดพลาด", "model": "", "ok": False,
                      "detail": repr(exc)[:300], "ms": 0})
    finally:
        api.close()

    return {"ok": bool(steps) and all(s["ok"] for s in steps), "steps": steps,
            "greeting": GREETING}
