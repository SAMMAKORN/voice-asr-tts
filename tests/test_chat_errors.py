"""เรียกโมเดลไม่สำเร็จแล้วต้องเห็นในช่องแชท ไม่ใช่แค่ในแผงบันทึกระบบ

เคสจริงที่ทำให้ต้องมีไฟล์นี้: CHAT_MODEL ใน .env ชี้ไปโมเดลที่บัญชีไม่มีสิทธิ์เรียก
ทุกเทิร์นจึงได้ HTTP 403 แต่ผู้ใช้เห็นแค่คำถามของตัวเองแล้วเงียบ — แยกไม่ออกเลยว่า
ระบบพังหรือ AI เลือกจะไม่ตอบ กว่าจะรู้ต้นเหตุก็ต้องไปเปิด session.jsonl อ่านเอง
"""
from __future__ import annotations

import threading

import pytest

from vc.api import ApiError, explain

pytestmark = pytest.mark.unit


def _fail(exc: Exception):
    def stream(*a, **k):
        raise exc
        yield ""        # pragma: no cover — ทำให้เป็น generator
    return stream


# ────────────────────────────────────────── 1. แปลข้อความผิดพลาดให้คนอ่านรู้เรื่อง
@pytest.mark.parametrize("raw,contains", [
    ('chat 403: {"error":{"message":"team not allowed"}}', "403"),
    ("chat 401: nope", "API_KEY"),
    ("chat 404: nope", "404"),
    ("chat 429: slow down", "429"),
    ("chat 502: bad gateway", "502"),
    ("tts เชื่อมต่อไม่สำเร็จ: ConnectError('getaddrinfo failed')", "API_BASE_URL"),
    ("chat สตรีมขาดกลางทาง: x", "ขาดกลางคัน"),
    ("chat เรียกไม่สำเร็จ: ReadTimeout()", "ไม่ตอบภายในเวลา"),
])
def test_an_error_is_explained_in_one_readable_sentence(raw, contains) -> None:
    said = explain(raw)
    assert contains in said
    assert "\n" not in said and len(said) < 120, "ยาวหรือหลายบรรทัด = ไม่เหมาะกับฟองแชท"
    assert "{" not in said, "JSON ดิบไม่ควรโผล่ในช่องแชท"


def test_an_unknown_error_still_gets_a_sentence() -> None:
    assert explain("อะไรสักอย่างที่ไม่เคยเจอ")


# ─────────────────────────────────────────── 2. ความล้มเหลวต้องโผล่ในช่องบทสนทนา
def test_a_failed_turn_shows_a_system_message_in_the_chat(web_session, outbox) -> None:
    web_session.api.chat_stream = _fail(ApiError("chat 403: no access"))
    web_session.respond(1, threading.Event(), "สวัสดี")

    roles = [m["role"] for m in outbox.of("begin")]
    assert "system" in roles, "ล้มเหลวแล้วช่องแชทเงียบสนิท ผู้ใช้ไม่รู้ว่าเกิดอะไรขึ้น"
    said = "".join(m["text"] for m in outbox.of("delta"))
    assert "403" in said


def test_the_full_error_still_goes_to_the_log_panel(web_session, outbox) -> None:
    """ฟองแชทได้ประโยคสั้น ส่วนตัวเต็มยังต้องอยู่ให้ไล่ปัญหาได้"""
    web_session.api.chat_stream = _fail(ApiError("chat 403: team not allowed"))
    web_session.respond(1, threading.Event(), "สวัสดี")

    logs = [m["text"] for m in outbox.of("log") if m.get("level") == "error"]
    assert any("team not allowed" in t for t in logs)


def test_a_failed_turn_is_not_remembered_as_an_answer(web_session) -> None:
    """ข้อความแจ้งเตือนไม่ใช่คำพูดของ AI — ห้ามเข้าไปอยู่ในประวัติที่ส่งให้โมเดล"""
    web_session.api.chat_stream = _fail(ApiError("chat 500: boom"))
    web_session.respond(1, threading.Event(), "สวัสดี")

    assert not [m for m in web_session.messages if m["role"] == "assistant"]


# ───────────────────────────────── 3. วันเวลาใน system prompt ต้องสดใหม่ทุกเทิร์น
def test_the_system_prompt_is_rebuilt_every_turn(web_session) -> None:
    """เซสชันที่เปิดค้างข้ามคืนเคยบอกโมเดลว่าตอนนี้คือเวลาที่เปิดหน้าเว็บ"""
    stale = "ตอนนี้คือวันจันทร์ที่ 1 มกราคม พ.ศ. 2500 เวลา 00:00 น."
    web_session.messages[0] = {"role": "system", "content": stale}
    head = web_session._history()[0]

    assert head["role"] == "system"
    assert stale not in head["content"], "ยังใช้ system prompt ที่เก็บไว้ตอนเปิด session"
    assert web_session.cfg.now().strftime("%H:%M") in head["content"]


# ──────────────── 4. ตัวแก้คำผิดฝั่งคำตอบต้องทิ้งร่องรอยไว้เหมือนฝั่ง ASR
def test_what_the_reply_corrector_changed_is_logged(web_session) -> None:
    """เคยเงียบสนิท — พอคำตอบออกมาแปลก ก็แยกไม่ออกว่าโมเดลเขียนมาแบบนั้นเองไหม"""
    import json

    from vc.thaispell import engine

    if engine() is None:
        pytest.skip("เครื่องนี้ไม่ได้ติดตั้ง PyThaiNLP")

    clock = web_session.cfg.supplied_phrases(web_session.cfg.now())[0]

    def stream(*a, **k):
        yield f"ตอนนี้{clock[:-4]}เพี้ยนแล้วครับ"

    web_session.api.chat_stream = stream
    web_session.respond(1, threading.Event(), "กี่โมงแล้ว")

    events = [json.loads(line) for line
              in web_session.log.jsonl.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    fixes = [e for e in events if e["type"] == "spell_fix"]
    assert all(e.get("side") in {"asr", "reply"} for e in fixes), (
        "ไม่ได้บอกว่าเป็นการแก้ฝั่งไหน แยกไม่ออกเวลาไล่ log")
