"""P2-6 / P2-15 ระดับเทิร์น — เทิร์นเดียวพังต้องไม่ล้ม session และต้องคืนสถานะเสมอ"""
from __future__ import annotations

import httpx
import numpy as np
import pytest

from vc.api import ApiError
from vc.phase import TurnPhase

pytestmark = pytest.mark.unit

PCM = np.zeros(12800, np.int16)          # 0.8 วินาทีที่ 16 kHz


def _speak(session, text: str = "ทดสอบเสียงพูดของผู้ใช้ครับ") -> None:
    session.api.transcribe = lambda pcm, sr: text      # type: ignore[method-assign]


def _record_events(session) -> list[tuple[str, dict]]:
    """ดักเหตุการณ์ที่ session บันทึกไว้ เพื่อยืนยันว่ามีการ log จริง"""
    seen: list[tuple[str, dict]] = []
    inner = session.log.event

    def event(kind: str, **fields) -> None:
        seen.append((kind, fields))
        inner(kind, **fields)

    session.log.event = event                          # type: ignore[method-assign]
    return seen


# ───────────────────────────────────────────── ทุกเส้นทางต้องเรียก _finish_epoch
@pytest.mark.parametrize("blow_up", ["transcribe", "chat"])
def test_finish_epoch_always_runs(web_session, blow_up) -> None:
    """AC-6.6 — ไม่ว่าพังที่ชั้นไหน สถานะต้องกลับเป็น IDLE เสมอ (spy)"""
    session = web_session
    session.cfg.tts_enabled = False
    calls: list[int] = []
    inner = session._finish_epoch
    session._finish_epoch = lambda: (calls.append(1), inner())[1]  # type: ignore[method-assign]

    def boom(*a, **k):
        raise RuntimeError("พังกลางเทิร์นแบบที่ไม่มีใครคาด")

    if blow_up == "transcribe":
        session.api.transcribe = boom                  # type: ignore[method-assign]
    else:
        _speak(session)
        session.api.chat_stream = boom                 # type: ignore[method-assign]

    session.handle_utterance(PCM)                      # ต้องไม่โยนออกมา

    assert calls, "_finish_epoch ไม่ถูกเรียกเลยเมื่อเทิร์นพัง"
    assert session.phase is TurnPhase.IDLE
    assert session.running.is_set(), "session ถูกปิดเพราะเทิร์นเดียวพัง"
    assert [m for m in session.out.of("log") if m.get("level") == "error"], (
        "ผู้ใช้ไม่ได้เห็นข้อความ error เลย")


def test_session_survives_api_error_and_answers_next_turn(web_session) -> None:
    """AC-6.1 — เน็ตกระตุกแล้วเทิร์นถัดไปต้องคุยต่อได้"""
    session = web_session
    session.cfg.tts_enabled = False
    fail = {"on": True}

    def transcribe(pcm, sr):
        if fail["on"]:
            raise ApiError("asr 503: ไม่พร้อมให้บริการ")
        return "ถามใหม่อีกครั้งนะครับ"

    def chat_stream(messages, cancel, **kw):
        yield "ได้เลยครับ"

    session.api.transcribe = transcribe                # type: ignore[method-assign]
    session.api.chat_stream = chat_stream              # type: ignore[method-assign]

    session.handle_utterance(PCM)
    assert session.phase is TurnPhase.IDLE
    fail["on"] = False
    session.handle_utterance(PCM)
    answers = [m["content"] for m in session.messages if m["role"] == "assistant"]
    assert answers == ["ได้เลยครับ"]


def test_raw_httpx_error_from_any_layer_is_contained(web_session) -> None:
    """แม้ชั้นล่างเผลอปล่อย httpx ออกมา เทิร์นก็ต้องไม่ล้ม session"""
    session = web_session
    session.cfg.tts_enabled = False
    _speak(session)

    def chat_stream(messages, cancel, **kw):
        yield "เริ่ม"
        raise httpx.ReadTimeout("หลุดกลางทาง")

    session.api.chat_stream = chat_stream              # type: ignore[method-assign]
    session.handle_utterance(PCM)
    assert session.phase is TurnPhase.IDLE
    assert session.running.is_set()


# ────────────────────────────────────────────────────── เพดานความยาวคำตอบในเทิร์นจริง
LONG_SENTENCE = "รายละเอียดเรื่องนี้ค่อนข้างยาวครับ ผมขออธิบายทีละส่วนให้ฟังนะครับ. "


def test_long_reply_is_capped_within_a_real_turn(web_session) -> None:
    """AC-15.1 + AC-15.2 — คำตอบยาวถูกตัด ไม่มี chunk ค้างในคิว และกลับสู่ IDLE"""
    session = web_session
    session.cfg.tts_enabled = True
    session.cfg.reply_max_chars = 320
    session.cfg.reply_max_sentences = 4
    session.speaker.LOST_GRACE = 0.2
    _speak(session)

    def chat_stream(messages, cancel, **kw):
        for _ in range(20):                            # ~1,300 ตัวอักษร
            if cancel.is_set():
                return
            yield LONG_SENTENCE

    session.api.chat_stream = chat_stream              # type: ignore[method-assign]
    session.api.synthesize = (                         # type: ignore[method-assign]
        lambda text, sr: np.zeros(int(session.cfg.speaker_sr * 0.02), np.int16))
    events = _record_events(session)
    session.start_workers()
    session.handle_utterance(PCM)

    answer = [m["content"] for m in session.messages if m["role"] == "assistant"][-1]
    assert len(answer) <= 320, f"คำตอบยังยาวเกินเพดาน ({len(answer)} ตัวอักษร)"
    assert session.phase is TurnPhase.IDLE
    assert session.tts_queue.empty(), "ยังมีงาน TTS ค้างในคิวหลังถูกตัด"
    assert session._inflight == 0, f"inflight ไม่กลับเป็นศูนย์ ({session._inflight})"
    assert [k for k, _ in events if k == "reply_capped"], (
        "ไม่ได้บันทึกเหตุการณ์ที่ถูกตัดไว้ให้วัดสัดส่วนภายหลัง")


def test_twenty_turns_never_exceed_the_cap(web_session) -> None:
    """AC-15.4 — 20 เทิร์นด้วยคำตอบยาวจริง สัดส่วนที่เกินเพดาน = 0%"""
    session = web_session
    session.cfg.tts_enabled = False
    session.cfg.reply_max_chars = 320
    _speak(session)

    def chat_stream(messages, cancel, **kw):
        text = LONG_SENTENCE * 12
        for i in range(0, len(text), 9):
            if cancel.is_set():
                return
            yield text[i:i + 9]

    session.api.chat_stream = chat_stream              # type: ignore[method-assign]
    over = 0
    for _ in range(20):
        session.handle_utterance(PCM)
        answer = [m["content"] for m in session.messages
                  if m["role"] == "assistant"][-1]
        if len(answer) > 320:
            over += 1
    assert over == 0
