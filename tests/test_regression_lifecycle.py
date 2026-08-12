"""เทสต์กันบั๊กเก่ากลับมา (regression) — งาน P1-0

สามข้อนี้เขียนขึ้น "ก่อน" การแก้ ทั้งสามจึงต้อง FAIL บนโค้ดเดิม และตอนนี้ GREEN
ครบทั้งสามแล้ว (marker `phase2` ถูกถอดออกเมื่องานที่เกี่ยวข้องเสร็จ):

    test_request_stop_terminates_all_threads            → GREEN ตั้งแต่ P1-2
    test_continued_speech_during_asr_still_answers       → GREEN เมื่อทำ P2-5
    test_transcribe_connect_error_does_not_kill_session  → GREEN เมื่อทำ P2-6

ทั้งหมดรันแบบไม่แตะเน็ตและไม่แตะอุปกรณ์เสียงจริง (ApiClient ถูกสวมทับด้วยของปลอม)
"""
from __future__ import annotations

import threading
import time

import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.unit

JOIN_TIMEOUT = 2.0        # AC-2.1: ทุกเธรดของ session ต้องจบภายใน 2 วินาที


# ───────────────────────────────────────────────────────────────────────── 1
def test_request_stop_terminates_all_threads(web_session) -> None:
    """ปิด session ระหว่างที่ยังมีงาน TTS ค้าง → ทุกเธรดต้องจบภายใน 2 วินาที

    บั๊กเดิม: web/session.py `request_stop()` เคลียร์แค่ `running` ไม่ได้ set
    `cancel` และไม่ล้างคิว TTS ทำให้ (ก) `_tts_worker` ออกจากลูปโดยทิ้ง item
    ค้างไว้พร้อม `inflight > 0` และ (ข) ลูปรอใน `greet()`/`respond()` ซึ่งเช็ค
    แค่ `_tts_busy()` วนไม่มีวันจบ → เธรด session ค้างตลอดอายุ process
    """
    session = web_session
    held = threading.Event()      # ปล่อยให้ TTS ก้อนแรกสังเคราะห์เสร็จเมื่อสั่ง

    def slow_synthesize(text: str, sr: int) -> np.ndarray:
        held.wait(JOIN_TIMEOUT)
        return np.zeros(240, np.int16)

    session.api.synthesize = slow_synthesize        # type: ignore[method-assign]
    session.speaker.LOST_GRACE = 0.2                # ไม่ต้องรอ client จริงรายงานกลับ

    session.start_workers()                         # เธรด tts
    worker = threading.Thread(target=session.greet, name="session", daemon=True)
    worker.start()

    # ยัดงาน TTS ค้างในคิวเพิ่ม (เลียนแบบผู้ใช้ปิดแท็บตอน AI กำลังพูดยาว ๆ)
    for seq in range(1, 4):
        session.enqueue_tts(0, seq, f"ประโยคที่ {seq} ที่ยังไม่ได้พูด")
    time.sleep(0.2)

    tts = next(t for t in threading.enumerate() if t.name == "tts")

    session.request_stop()
    held.set()

    worker.join(JOIN_TIMEOUT)
    # AC-2.1 พูดถึง "ทุกเธรดของ session" — เธรด tts ก็ต้องจบด้วย และ invariant
    # เรื่อง inflight เป็นจริงหลังเธรดนี้จบ (ดู `_tts_worker`) จึงต้องรอให้จบก่อนวัด
    tts.join(JOIN_TIMEOUT)
    alive = [t.name for t in threading.enumerate()
             if t.name in ("session", "tts") and t.is_alive()]

    assert not worker.is_alive(), (
        "เธรด session ยังไม่จบหลัง request_stop() — ลูปรอ TTS วนไม่จบ "
        f"(เธรดที่ยังค้าง: {alive})")
    assert not tts.is_alive(), f"เธรด tts ยังไม่จบหลัง request_stop() ({alive})"
    assert session._inflight == 0, (      # AC-2.2
        f"ยังมีงาน TTS ค้างนับไว้ inflight={session._inflight}")
    assert session.tts_queue.empty(), "คิว TTS ไม่ถูกล้างตอนปิด session"


# ───────────────────────────────────────────────────────────────────────── 2
def test_continued_speech_during_asr_still_answers(web_session) -> None:
    """พูดต่ออีกประโยคระหว่างระบบกำลังถอดเสียง → ต้องได้คำตอบ 1 คำตอบ

    บั๊กเดิม: `busy` เป็น bool ตัวเดียวคุมทั้ง "ถอดเสียง" และ "พูด"
    `_on_speech_start()` จึงเห็น busy=True แล้วสั่ง interrupt ตัวเอง →
    `cancel` ถูก set ก่อนเข้า `respond()` → เทิร์นหายเงียบ ผู้ใช้ไม่ได้คำตอบเลย
    """
    session = web_session
    session.cfg.tts_enabled = False        # ทดสอบเส้นทางคำตอบล้วน ไม่ต้องมีเสียง

    said = ["ประโยคแรกของผู้ใช้", "ประโยคที่สองที่พูดต่อ"]
    calls: list[int] = []

    def transcribe(pcm: np.ndarray, sr: int) -> str:
        calls.append(1)
        time.sleep(0.2)                    # ช่วงที่ระบบ "กำลังถอดเสียง"
        return said[min(len(calls) - 1, len(said) - 1)]

    def chat_stream(messages, cancel, **kwargs):
        for token in ("คำ", "ตอบ", "จากเอไอ"):
            if cancel.is_set():
                return
            yield token

    session.api.transcribe = transcribe            # type: ignore[method-assign]
    session.api.chat_stream = chat_stream          # type: ignore[method-assign]

    pcm = np.zeros(int(session.cfg.mic_sr * 0.8), np.int16)
    turn = threading.Thread(target=session.handle_utterance, args=(pcm,), daemon=True)
    turn.start()

    time.sleep(0.05)                               # ตอนนี้อยู่ระหว่างถอดเสียง
    session._on_speech_start()                     # ผู้ใช้พูดต่อ
    session.events.put(("utterance", pcm))
    turn.join(5.0)

    assert not turn.is_alive(), "เทิร์นค้างไม่จบ"
    users = [m["content"] for m in session.messages if m["role"] == "user"]
    answers = [m["content"] for m in session.messages if m["role"] == "assistant"]

    assert len(calls) == 2, f"ต้องถอดเสียงทั้งสองประโยค (ถอดจริง {len(calls)} ครั้ง)"
    assert len(users) == 1 and all(s in users[0] for s in said), (
        f"สองประโยคต้องรวมเป็นเทิร์นเดียว ได้: {users}")
    assert answers == ["คำตอบจากเอไอ"], (
        "เทิร์นถูกยกเลิกเงียบ ๆ — ผู้ใช้ไม่ได้คำตอบ (ได้: %r)" % (answers,))


# ───────────────────────────────────────────────────────────────────────── 3
def test_transcribe_connect_error_does_not_kill_session(web_session) -> None:
    """เน็ตกระตุกตอนถอดเสียง → session ต้องอยู่ต่อและพูดเทิร์นถัดไปได้

    บั๊กเดิม: `handle_utterance` ดักแค่ `ApiError` ส่วน `httpx.ConnectError`
    ทะลุขึ้นไปถึง `event_loop()` → `run()` → ปิดทั้ง session และ `busy` ค้าง True
    เพราะ `_finish_epoch()` ไม่เคยถูกเรียก
    """
    session = web_session
    session.cfg.tts_enabled = False
    fail = {"on": True}

    def transcribe(pcm: np.ndarray, sr: int) -> str:
        if fail["on"]:
            raise httpx.ConnectError("เชื่อมต่อไม่ได้ (จำลองเน็ตหลุด)")
        return "ถามใหม่อีกครั้ง"

    def chat_stream(messages, cancel, **kwargs):
        yield "ได้ครับ"

    session.api.transcribe = transcribe            # type: ignore[method-assign]
    session.api.chat_stream = chat_stream          # type: ignore[method-assign]

    pcm = np.zeros(int(session.cfg.mic_sr * 0.8), np.int16)
    session.handle_utterance(pcm)                  # ต้องไม่โยน exception ออกมา

    assert session.busy is False, "เทิร์นจบแล้วแต่สถานะยังค้างว่ากำลังยุ่ง"
    assert session.running.is_set(), "session ถูกปิดทั้งที่แค่เน็ตกระตุกชั่วคราว"

    errors = [m for m in session.out.of("log") if m.get("level") == "error"]
    assert errors, "ไม่มีข้อความบอกผู้ใช้ว่าถอดเสียงไม่สำเร็จ"

    fail["on"] = False
    session.handle_utterance(pcm)                  # เทิร์นถัดไปต้องสำเร็จ
    answers = [m["content"] for m in session.messages if m["role"] == "assistant"]
    assert answers == ["ได้ครับ"], f"เทิร์นถัดไปยังตอบไม่ได้ (ได้: {answers})"
