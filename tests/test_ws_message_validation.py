"""P2-11 — ข้อความผิดรูปจากเบราว์เซอร์ต้องถูกข้าม ไม่ใช่ปิดทั้ง session

เดิม `int(msg.get("epoch"))` รับ string/None/dict แล้วโยน exception ขึ้นไปถึงลูป
transport ซึ่ง `finally` ปิด session ทิ้ง — client ที่ซนหรือแค่บั๊กเล็ก ๆ ก็ตัดสายได้
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from voice_chat import EVENTS_MAXSIZE
from web.session import MAX_AUDIO_BYTES, MAX_TEXT_CHARS

pytestmark = pytest.mark.unit


def alive(session) -> bool:
    return session.running.is_set() and not session._closed.is_set()


# ────────────────────────────────────────────────────────── field ที่ผิดชนิด
def test_bad_epoch_does_not_close_the_session(web_session) -> None:
    """AC-11.1 — epoch เป็นสตริงมั่ว ๆ แล้วข้อความถัดไปต้องยังทำงานปกติ"""
    session = web_session
    session.handle_client({"type": "audio", "epoch": "abc"})
    session.handle_client({"type": "played", "epoch": "abc", "seq": None})
    assert alive(session), "session ถูกปิดเพราะข้อความเดียวผิดรูป"
    assert session.bad_messages >= 2

    session.handle_client({"type": "text", "text": "ข้อความที่ถูกต้อง"})
    assert ("text", "ข้อความที่ถูกต้อง") in list(session.events.queue)


def test_played_and_level_only_accept_usable_numbers(web_session) -> None:
    session = web_session
    session.speaker.set_epoch(2)
    session.speaker.play(np.ones(240, np.int16), tag=(2, 5, "ก"), epoch=2)

    for msg in ({"type": "played", "epoch": {"x": 1}, "seq": 5},
                {"type": "played", "epoch": -3, "seq": 5},
                {"type": "level", "out": "ดังมาก"},
                {"type": "level", "out": float("nan")},
                {"type": "level", "out": None}):
        session.handle_client(msg)
    assert alive(session)
    assert session.speaker.finished_tags == [], "รับ epoch/seq ที่ผิดรูปไปใช้จริง"

    session.handle_client({"type": "played", "epoch": "2", "seq": "5"})
    assert [t[1] for t in session.speaker.finished_tags] == [5], (
        "ตัวเลขที่ส่งมาเป็นสตริงยังต้องใช้งานได้ตามเดิม")


# ──────────────────────────────────────────────────────────────── เสียงดิบ
def test_odd_sized_audio_frame_is_dropped(web_session) -> None:
    """AC-11.2 — จำนวนไบต์เป็นเลขคี่ ต้องทิ้งเฟรมพร้อม log ไม่ใช่ crash"""
    session = web_session
    before = session.mic.received
    session.feed_audio(b"\x01\x02\x03")
    assert session.mic.received == before, "เฟรมเลขคี่ถูกรับเข้าไปประมวลผล"
    assert session.bad_messages == 1
    assert alive(session)

    session.feed_audio(np.zeros(320, np.int16).tobytes())
    assert session.mic.received == before + 320


def test_oversized_audio_frame_is_dropped(web_session) -> None:
    session = web_session
    session.feed_audio(b"\x00" * (MAX_AUDIO_BYTES + 2))
    assert session.mic.received == 0
    assert session.bad_messages == 1
    assert alive(session)


# ───────────────────────────────────────────────────────────── ข้อความยาว
def test_huge_text_is_truncated_not_accepted_whole(web_session) -> None:
    """AC-11.3 — text ยาว 10 MB ต้องถูกตัด และหน่วยความจำไม่พุ่ง"""
    session = web_session
    session.handle_client({"type": "text", "text": "ก" * 10_000_000})
    queued = [payload for kind, payload in list(session.events.queue)
              if kind == "text"]
    assert queued and len(queued[0]) <= MAX_TEXT_CHARS
    assert session.bad_messages == 1
    assert alive(session)


def test_text_of_wrong_type_is_ignored(web_session) -> None:
    session = web_session
    for value in ({"a": 1}, ["x"], 5, None):
        session.handle_client({"type": "text", "text": value})
    assert list(session.events.queue) == []
    assert alive(session)


# ──────────────────────────────────────────────────────── คิวมีเพดาน ไม่โตไม่จำกัด
def test_event_queue_is_bounded_and_drops_oldest(web_session) -> None:
    """AC-11.4 — ยิงเร็วกว่าที่ consumer ดึงได้ คิวต้องหยุดโตที่ maxsize"""
    session = web_session
    for i in range(EVENTS_MAXSIZE * 10):
        session.handle_client({"type": "text", "text": f"ข้อความที่ {i}"})
    assert session.events.qsize() == EVENTS_MAXSIZE
    assert session.dropped_events > 0
    # ของใหม่สุดต้องยังอยู่ (นโยบายทิ้งของเก่าสุด)
    newest = list(session.events.queue)[-1]
    assert newest == ("text", f"ข้อความที่ {EVENTS_MAXSIZE * 10 - 1}")
    assert alive(session)


def test_quit_still_gets_through_a_full_queue(web_session) -> None:
    session = web_session
    for i in range(EVENTS_MAXSIZE * 2):
        session.put_event("text", f"ข้อความ {i}")
    session.request_stop()
    assert ("quit", None) in list(session.events.queue), (
        "คำสั่งปิดหายไปเพราะคิวเต็ม — session จะไม่มีวันปิด")


# ─────────────────────────────────────────────────────────────────── fuzz
def _fuzz_payloads() -> list[dict]:
    """100 แบบจากส่วนผสมของ type/field ที่ผิดรูปทุกทรง"""
    types = ["played", "level", "text", "mute", "echo_guard", "interrupt",
             "clear", "quit", "audio", "", "ระเบิด", None, 7, {"a": 1}]
    values = [None, "", "abc", -1, 10 ** 20, float("inf"), float("nan"),
             {"nested": {"deep": [1, 2]}}, [1, 2, 3], True, "ก" * 5000]
    out: list[dict] = []
    for kind, value in itertools.product(types, values):
        out.append({"type": kind, "epoch": value, "seq": value,
                    "out": value, "text": value, "on": value})
    return out[:100]


def test_hundred_malformed_payloads_keep_the_session_open(web_session) -> None:
    """AC-11.5 — fuzz 100 แบบแล้ว session ต้องยังเปิดอยู่ทุกกรณี"""
    session = web_session
    payloads = _fuzz_payloads()
    assert len(payloads) == 100
    for payload in payloads:
        if payload.get("type") == "quit":
            continue                     # quit เป็นคำสั่งที่ถูกต้อง ไม่ใช่ข้อความผิดรูป
        session.handle_client(payload)
        assert alive(session), f"session ปิดเพราะ payload: {payload!r}"

    for raw in (b"", b"\x00", b"\x01" * 7, b"\xff" * (MAX_AUDIO_BYTES + 1)):
        session.feed_audio(raw)
        assert alive(session), f"session ปิดเพราะเฟรมเสียง {len(raw)} ไบต์"
