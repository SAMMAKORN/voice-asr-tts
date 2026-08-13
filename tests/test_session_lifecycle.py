"""P1-2 — ปิด session แล้วต้องไม่มีเธรดรั่ว ไม่มี busy-loop และปิดทรัพยากรเสมอ"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
import pytest

from conftest import FakeOutbox
from web import server
from web.session import WebSession

pytestmark = pytest.mark.unit


def _start_speaking(session, held: threading.Event) -> threading.Thread:
    """จำลอง 'AI กำลังพูด': เธรด tts ทำงานอยู่ + เธรด session รอเสียงจบ"""
    session.api.synthesize = lambda text, sr: (held.wait(2.0),
                                               np.zeros(240, np.int16))[1]
    session.speaker.LOST_GRACE = 0.2
    session.start_workers()
    worker = threading.Thread(target=session.greet, name="session", daemon=True)
    worker.start()
    for seq in range(1, 4):
        session.enqueue_tts(0, seq, f"ก้อนที่ {seq}")
    time.sleep(0.2)
    return worker


# ───────────────────────────────────────────────────────────────────── AC-2.3
def test_no_busy_loop_after_close(web_session) -> None:
    """ปิดแท็บระหว่าง AI พูด → ลูปรอต้องหยุดวน ไม่ใช่ spin 33 ครั้ง/วินาทีต่อไป"""
    session = web_session
    held = threading.Event()
    worker = _start_speaking(session, held)

    session.request_stop()
    held.set()
    worker.join(2.0)
    assert not worker.is_alive()

    spins = session._wait_spins
    time.sleep(0.5)          # ถ้ายังวนอยู่ 0.5 วิ = ราว 16 รอบ
    assert session._wait_spins - spins < 5, (
        f"ลูปรอยังวนต่อหลังปิด session ({session._wait_spins - spins} รอบใน 0.5 วินาที)")


# ───────────────────────────────────────────────────────────────────── AC-2.4
def test_join_timeout_logs_error_and_still_closes_resources(monkeypatch, caplog) -> None:
    """เธรด session ค้างจน join ไม่สำเร็จ → ต้อง log ERROR และปิดทรัพยากรให้อยู่ดี"""
    stuck = threading.Event()
    closed: list[str] = []

    class StuckSession:
        def request_stop(self) -> None:
            closed.append("request_stop")

        def close_resources(self) -> None:
            closed.append("close_resources")

    worker = threading.Thread(target=lambda: stuck.wait(5.0),
                              name="session", daemon=True)
    worker.start()
    monkeypatch.setattr(server, "JOIN_TIMEOUT", 0.2)

    with caplog.at_level(logging.ERROR, logger="voicechat.web"):
        asyncio.run(server.close_session(StuckSession(), worker))
    stuck.set()
    worker.join(2.0)

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("join timeout" in m and "session" in m for m in errors), errors
    assert closed == ["request_stop", "close_resources"], closed


def test_close_resources_is_idempotent_and_closes_log_and_api(web_session) -> None:
    calls: list[str] = []
    web_session.log.close = lambda *a, **k: calls.append("log")
    web_session.api.close = lambda: calls.append("api")

    web_session.close_resources()
    web_session.close_resources()
    assert calls == ["log", "api"]


def test_close_resources_survives_a_failing_closer(web_session) -> None:
    def boom() -> None:
        raise RuntimeError("ปิดไฟล์ไม่ได้")

    web_session.log.close = boom
    api_closed: list[bool] = []
    web_session.api.close = lambda: api_closed.append(True)

    web_session.close_resources()          # ต้องไม่โยนต่อ
    assert api_closed == [True], "log พังแล้วข้ามการปิด api ไปเลย"


# ───────────────────────────────────────────────────────────────────── AC-2.5
@pytest.mark.slow
def test_twenty_sessions_do_not_leak_threads(cfg) -> None:
    baseline = threading.active_count()
    for _ in range(20):
        session = WebSession(cfg, FakeOutbox())          # type: ignore[arg-type]
        held = threading.Event()
        worker = _start_speaking(session, held)
        session.request_stop()
        held.set()
        worker.join(2.0)
        session.close_resources()
        assert not worker.is_alive()

    for _ in range(40):        # เธรด tts ใช้ timeout 0.2 วินาที ให้เวลามันปิดตัว
        if threading.active_count() <= baseline + 1:
            break
        time.sleep(0.1)
    assert threading.active_count() <= baseline + 1, (
        f"เธรดสะสม: เริ่ม {baseline} จบ {threading.active_count()} "
        f"({[t.name for t in threading.enumerate()]})")
