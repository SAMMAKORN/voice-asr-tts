"""P2-5 — แยกช่วงของเทิร์นเป็น TurnPhase แทนธง busy ตัวเดียว

พิสูจน์กติกาพูดแทรกใหม่:
  TRANSCRIBING = ไม่ขัด (ต่อประโยคเดิม) · GENERATING/SPEAKING = ขัดได้จริง
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from vc.phase import BARGE_IN_PHASES, TurnPhase, interrupts

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def _tokens(*words: str):
    def chat_stream(messages, cancel, **kwargs):
        for w in words:
            if cancel.is_set():
                return
            yield w
    return chat_stream


# ───────────────────────────────────────────────────── ตารางกติกาพูดแทรก (P2-5)
def test_barge_in_table() -> None:
    assert not interrupts(TurnPhase.IDLE)
    assert not interrupts(TurnPhase.TRANSCRIBING), "ช่วงถอดเสียงห้ามนับเป็นพูดแทรก"
    assert interrupts(TurnPhase.GENERATING)
    assert interrupts(TurnPhase.SPEAKING)
    assert BARGE_IN_PHASES == {TurnPhase.GENERATING, TurnPhase.SPEAKING}


def test_speech_during_transcribing_does_not_interrupt(web_session) -> None:
    """AC-5.1 (ระดับหน่วย) — phase = TRANSCRIBING แล้วมีเสียงเข้ามาต้องไม่ตั้ง cancel"""
    session = web_session
    session._set_phase(TurnPhase.TRANSCRIBING)
    session._on_speech_start()
    assert not session.cancel.is_set(), "ขัดจังหวะตัวเองระหว่างถอดเสียง — เทิร์นจะหาย"
    assert session._barged is False


@pytest.mark.parametrize("phase", [TurnPhase.GENERATING, TurnPhase.SPEAKING])
def test_speech_while_ai_holds_the_mic_interrupts(web_session, phase) -> None:
    """AC-5.3 — ช่วงคิด/พูด เสียงผู้ใช้ต้องยกเลิกเทิร์นเดิมทันที"""
    session = web_session
    session._set_phase(phase)
    session._on_speech_start()
    assert session.cancel.is_set()
    assert session._barged is True
    assert session.out.of("stop"), "ไม่ได้สั่งเบราว์เซอร์หยุดเล่นเสียง"


# ─────────────────────────────────────────────────────────── คำทักทายพูดแทรกได้
def test_greeting_is_interruptible(web_session) -> None:
    """AC-5.2 — พูดแทรกคำทักทายได้จริง (เดิม greet() อยู่นอก epoch → แทรกไม่ได้)"""
    session = web_session
    session.speaker.LOST_GRACE = 0.3
    synth = threading.Event()

    def slow_synthesize(text: str, sr: int) -> np.ndarray:
        synth.set()
        return np.zeros(int(session.cfg.speaker_sr * 0.5), np.int16)

    session.api.synthesize = slow_synthesize     # type: ignore[method-assign]
    session.start_workers()
    greeting = threading.Thread(target=session.greet, name="greet", daemon=True)
    greeting.start()

    assert synth.wait(2.0), "คำทักทายไม่ถูกส่งเข้าคิวเสียงเลย"
    deadline = time.monotonic() + 1.0
    while session.phase is not TurnPhase.SPEAKING and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.phase is TurnPhase.SPEAKING, (
        f"ระหว่างทักทายต้องอยู่ในช่วง SPEAKING (ได้ {session.phase})")

    t0 = time.monotonic()
    session._on_speech_start()                   # ผู้ใช้พูดแทรกคำทักทาย
    assert session.out.of("stop"), "เสียงไม่ถูกสั่งหยุด"
    assert (time.monotonic() - t0) < 0.3, "สั่งหยุดเสียงช้ากว่า 300ms"

    greeting.join(2.0)
    assert not greeting.is_alive()
    assert session.phase is TurnPhase.IDLE

    # เทิร์นของผู้ใช้หลังพูดแทรกต้องถูกประมวลผลจริง
    session.cfg.tts_enabled = False
    session.api.transcribe = lambda pcm, sr: "ช่วยบอกเวลาตอนนี้ให้ด้วยครับ"  # type: ignore[method-assign]
    session.api.chat_stream = _tokens("ได้", "ครับ")   # type: ignore[method-assign]
    session.handle_utterance(np.zeros(int(session.cfg.mic_sr * 0.8), np.int16))
    answers = [m["content"] for m in session.messages if m["role"] == "assistant"]
    assert answers[-1] == "ได้ครับ", f"เทิร์นหลังพูดแทรกไม่ได้คำตอบ ({answers})"


# ──────────────────────────────────────────────────────────── log ของ transition
def test_phase_transitions_are_logged(web_session, caplog) -> None:
    """AC-5.5 — เห็น IDLE→TRANSCRIBING→GENERATING→SPEAKING→IDLE ครบใน log debug"""
    session = web_session
    session.speaker.LOST_GRACE = 0.2
    session.api.transcribe = lambda pcm, sr: "วันนี้อากาศเป็นอย่างไรครับ"  # type: ignore[method-assign]
    session.api.chat_stream = _tokens("อากาศดีครับ")   # type: ignore[method-assign]
    session.api.synthesize = (                          # type: ignore[method-assign]
        lambda text, sr: np.zeros(int(session.cfg.speaker_sr * 0.05), np.int16))
    session.start_workers()

    with caplog.at_level(logging.DEBUG, logger="voicechat.phase"):
        session.handle_utterance(np.zeros(int(session.cfg.mic_sr * 0.8), np.int16))

    seen = [re.search(r"phase (\w+)→(\w+)", r.message) for r in caplog.records]
    chain = [(m.group(1), m.group(2)) for m in seen if m]
    assert ("idle", "transcribing") in chain
    assert ("transcribing", "generating") in chain
    assert ("generating", "speaking") in chain, f"ไม่เห็นช่วงพูด: {chain}"
    assert chain[-1][1] == "idle", f"เทิร์นจบแล้วไม่กลับสู่ idle: {chain}"

    # phase จริงถูกส่งขึ้นไปให้ UI ด้วย (AC-5.3 ท่อน "UI แสดง phase ตรงกับความจริง")
    values = [m["value"] for m in session.out.of("phase")]
    assert values[:3] == ["transcribing", "generating", "speaking"]
    assert values[-1] == "idle"


# ────────────────────────────────────────────────────── ไม่เหลือ busy คุม barge-in
def test_no_busy_flag_drives_barge_in() -> None:
    """AC-5.4 — `busy` เหลือได้แค่ property เพื่อความเข้ากันได้ ห้ามใช้ตัดสินใจพูดแทรก"""
    src = (ROOT / "voice_chat.py").read_text(encoding="utf-8")
    assert "self.busy = " not in src, "ยังมีการเขียนค่าใส่ธง busy อยู่"
    assert "def busy(self)" in src, "ควรเหลือ busy เป็น property (backward compat)"

    # จุดตัดสินใจพูดแทรกต้องอ่าน phase ไม่ใช่ busy (คอมเมนต์อธิบายมีคำว่า busy ได้)
    start = src.index("def _on_speech_start")
    body = src[start:src.index("def _on_utterance", start)]
    assert "BARGE_IN_PHASES" in body
    assert "self.busy" not in body
