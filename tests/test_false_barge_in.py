"""P3-23 — การยืนยันการพูดแทรกจริง (ทางเลือก A) + เก็บกวาด dead code

ที่เลือกทาง A เพราะข้อมูลจริงบอกว่าปัญหามีอยู่จริงและแก้ได้: ในบันทึก 37 session
มี ASR ตามหลังการพูดแทรก 34 ครั้ง ในนั้นเป็นเสียงสะท้อนชัด ๆ 5 ครั้ง (15%)
* `啥东西？` `我爱你。` `bản thân cậu.` — เกณฑ์ "ไม่มีอักขระไทย" จับได้
* `ครับ` × 2 — เกณฑ์เดิมจับไม่ได้เลย ต้องเทียบกับสิ่งที่ AI เพิ่งพูด (ของใหม่)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vc.echo import (ECHO_OF_REPLY, TOO_SHORT, WRONG_LANGUAGE, ECHO_MAX_CHARS,
                     noise_reason, normalise)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent

# ข้อความจริงจากบันทึก (logs/) ที่ตามหลังเหตุการณ์ "ผู้ใช้พูดแทรก"
REAL_ECHOES = ["啥东西？", "我爱你。", "bản thân cậu.", "ครับ", "ครับ"]
REAL_REPLY_TAIL = "เดี๋ยวผมสรุปให้สั้น ๆ นะครับ"
REAL_SPEECH = [
    "ช่วยสรุปให้สั้นกว่านี้หน่อยได้ไหม",
    "เดี๋ยวก่อนครับ ผมอยากถามอีกเรื่อง",
    "หยุด",
    "แล้วราคาทองวันนี้เท่าไร",
]


# ────────────────────────────────────────────────────── เกณฑ์แต่ละข้อ (AC-23.3)
def test_too_short_is_not_speech() -> None:
    assert noise_reason("", barged=True) == TOO_SHORT
    assert noise_reason("ก", min_chars=2, barged=True) == TOO_SHORT


def test_short_non_thai_while_speaking_is_echo() -> None:
    for text in ("啥东西？", "我爱你。", "bản thân cậu."):
        assert noise_reason(text, REAL_REPLY_TAIL, barged=True) == WRONG_LANGUAGE


def test_a_word_the_ai_just_said_is_echo() -> None:
    """ของใหม่ในทาง A — เคส `ครับ` ที่เกณฑ์เดิมปล่อยผ่านทุกครั้ง"""
    assert noise_reason("ครับ", REAL_REPLY_TAIL, barged=True) == ECHO_OF_REPLY
    assert noise_reason("นะครับ", REAL_REPLY_TAIL, barged=True) == ECHO_OF_REPLY
    assert noise_reason("ครับ.", REAL_REPLY_TAIL, barged=True) == ECHO_OF_REPLY


def test_every_real_echo_from_the_logs_is_caught() -> None:
    """AC-23.3 — ต้อง trigger จริงกับข้อมูลจริง ไม่ใช่แค่ทฤษฎี"""
    caught = [t for t in REAL_ECHOES
              if noise_reason(t, REAL_REPLY_TAIL, barged=True)]
    assert caught == REAL_ECHOES, f"ยังจับไม่ได้: {set(REAL_ECHOES) - set(caught)}"


def test_real_speech_is_never_treated_as_echo() -> None:
    """สำคัญกว่าข้อบน: ห้ามกินคำพูดจริงของผู้ใช้"""
    for text in REAL_SPEECH:
        assert noise_reason(text, REAL_REPLY_TAIL, barged=True) is None, text


def test_a_long_word_repeated_from_the_reply_still_counts_as_speech() -> None:
    """ผู้ใช้ทวนคำที่ AI เพิ่งถามถือเป็นการพูดจริง — เกณฑ์เทียบเสียงจึงจำกัดความยาว"""
    spoken = "อยากฟังรายละเอียดต่อไหมครับ"
    assert len("รายละเอียด") > ECHO_MAX_CHARS
    assert noise_reason("รายละเอียด", spoken, barged=True) is None


def test_echo_rules_only_apply_while_the_ai_was_speaking() -> None:
    """ไม่ได้ถูกพูดแทรก = ไม่มีเสียงให้สะท้อน ห้ามตัดสินว่าเป็นเสียงสะท้อน"""
    assert noise_reason("ครับ", REAL_REPLY_TAIL, barged=False) is None
    assert noise_reason("啥东西？", REAL_REPLY_TAIL, barged=False) is None


def test_nothing_spoken_yet_means_no_echo_match() -> None:
    assert noise_reason("ครับ", "", barged=True) is None


def test_normalise_ignores_punctuation_and_spacing() -> None:
    assert normalise(" ครับ! ") == normalise("ครับ")
    assert normalise("") == ""


# ─────────────────────────────────────── ทำงานจริงในเทิร์น + ตัวนับ (AC-23.3)
def _turn(session, text: str, *, barged: bool) -> None:
    session.api.transcribe = lambda pcm, sr: text     # type: ignore[method-assign]
    session._barged = barged
    session.handle_utterance(np.zeros(320, np.int16))


def test_false_barge_in_counter_and_event(web_session) -> None:
    """AC-23.3 — assert counter ตามที่ AC สั่ง"""
    web_session._last_spoken = REAL_REPLY_TAIL
    web_session.messages.append({"role": "assistant", "content": "ตอบครึ่งเดียว"})
    web_session._last_reply = {"idx": len(web_session.messages) - 1,
                               "message": web_session.messages[-1],
                               "full": "ตอบครึ่งเดียวแต่ยาวกว่านั้น"}
    _turn(web_session, "ครับ", barged=True)
    assert web_session.false_barge_ins == 1
    # คำตอบเดิมถูกคืนเข้าประวัติ ไม่ถูกทิ้งเพราะเสียงสะท้อน
    assert web_session.messages[-1]["content"] == "ตอบครึ่งเดียวแต่ยาวกว่านั้น"


def test_real_interruption_is_not_counted(web_session, monkeypatch) -> None:
    web_session._last_spoken = REAL_REPLY_TAIL
    calls: list[str] = []
    monkeypatch.setattr(web_session, "respond",
                        lambda epoch, cancel, text: calls.append(text))
    _turn(web_session, "เดี๋ยวก่อนครับ ผมอยากถามอีกเรื่อง", barged=True)
    assert web_session.false_barge_ins == 0
    assert calls == ["เดี๋ยวก่อนครับ ผมอยากถามอีกเรื่อง"]


def test_reason_is_recorded_in_the_log(web_session) -> None:
    web_session._last_spoken = REAL_REPLY_TAIL
    _turn(web_session, "我爱你。", barged=True)
    events = [json.loads(line) for line in
              web_session.log.jsonl.read_text(encoding="utf-8").splitlines() if line]
    hit = [e for e in events if e["type"] == "false_barge_in"]
    assert hit and hit[0]["reason"] == WRONG_LANGUAGE
    assert hit[0]["total"] == 1


# ──────────────────────────────────────────── mic telemetry ที่เคยนับแล้วทิ้ง (AC-23.1)
def test_input_stats_reach_the_log(web_session) -> None:
    """AC-23.1 — overflows/clipped/dropped ต้องปรากฏในสรุปตอนจบ session"""
    web_session.mic.dropped = 3
    web_session.dropped_events = 2
    web_session.log_input_stats()
    events = [json.loads(line) for line in
              web_session.log.jsonl.read_text(encoding="utf-8").splitlines() if line]
    stats = [e for e in events if e["type"] == "input_stats"]
    assert stats, "ไม่มีเหตุการณ์ input_stats เลย"
    for key in ("overflows", "clipped", "dropped_frames", "dropped_events",
                "false_barge_ins"):
        assert key in stats[0], key
    assert stats[0]["dropped_frames"] == 3 and stats[0]["dropped_events"] == 2


def test_microphone_counts_dropped_frames() -> None:
    from vc.audio import Microphone

    mic = Microphone(16000, 20)
    mic.frames = __import__("queue").Queue(maxsize=1)
    frame = np.zeros((320, 1), np.int16)
    mic._callback(frame, 320, None, None)
    mic._callback(frame, 320, None, None)
    assert mic.dropped == 1


# ────────────────────────────────────────────────────────── dead code (AC-23.2)
# ต้องระบุโฟลเดอร์ให้ชัด ห้ามใช้ ROOT.rglob() — ไม่งั้นบนเครื่องที่ทำตาม README
# (`python3 -m venv .venv`) จะกวาด .venv/lib/.../site-packages เข้ามาเป็นพันไฟล์
# แล้วฟ้อง dead code ของไลบรารีคนอื่น ส่วนบน path ที่มี `.claude` จะได้ลิสต์ว่าง
# แล้วผ่านแบบไม่ได้ตรวจอะไรเลย (เหมือน tests/test_layering.py ที่ทำถูกอยู่แล้ว)
SOURCES = ([p for folder in ("web", "vc") for p in (ROOT / folder).rglob("*.py")]
           + [ROOT / "voice_chat.py"])


def test_the_source_list_is_not_empty() -> None:
    """กันเทสต์ข้างล่างผ่านเพราะไม่มีไฟล์ให้ตรวจ"""
    assert SOURCES, "ไม่พบไฟล์ต้นฉบับให้ตรวจเลย"
    assert (ROOT / "vc" / "chat.py") in SOURCES


# `def sources(` ไม่อยู่ในลิสต์นี้: vc/websearch.py:sources() ถูกเรียกจริงจาก
# tests/test_search.py (คืนกลับมาใน 6641a42) จึงไม่ใช่ dead code อีกแล้ว
@pytest.mark.parametrize("pattern", ["\\.paused", "queued_seconds"])
def test_dead_code_is_gone(pattern: str) -> None:
    """AC-23.2 — ของที่ไม่มีผู้เรียกต้องไม่เหลืออยู่"""
    import re

    assert SOURCES
    hits = [f"{p.relative_to(ROOT)}:{i}"
            for p in SOURCES
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(pattern, line)]
    assert hits == [], f"ยังเหลือ dead code: {hits}"


@pytest.mark.parametrize("pattern", ["\\.paused", "queued_seconds"])
def test_the_dead_code_patterns_are_valid_regexes(pattern: str) -> None:
    """`def sources(` เคยหลุดเข้าลิสต์ทั้งที่ compile ไม่ผ่าน (วงเล็บไม่ได้ escape)"""
    import re

    re.compile(pattern)


def test_unused_colour_import_is_gone() -> None:
    """GRAY ถูก import มาโดยไม่มีใครใช้ (คลาสหลักย้ายไป vc/chat.py แล้วใน P3-21)"""
    lines = (ROOT / "vc" / "chat.py").read_text(encoding="utf-8").splitlines()
    line = [l for l in lines if l.startswith("from .ui import")][0]
    assert "GRAY" not in line
