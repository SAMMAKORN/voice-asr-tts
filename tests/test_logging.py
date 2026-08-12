"""P3-18 — บันทึกการสนทนา: ที่เก็บ, การหมดอายุ, สิทธิ์ไฟล์, สวิตช์ปิด transcript

บันทึกเป็นคำพูดจริงของผู้ใช้แบบ plaintext ถาวร จึงต้องตรวจสามเรื่องแบบอัตโนมัติ:
ตั้งที่เก็บได้จริงไหม · ลบของเก่าตามที่สั่งไหม · คนอื่นบนเครื่องอ่านได้หรือเปล่า
"""
from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from vc.config import now as tz_now
from vc.logger import (DIR_MODE, FILE_MODE, SessionLogger, purge_old_sessions,
                       session_started_at)

pytestmark = pytest.mark.unit

WINDOWS = sys.platform.startswith("win")


def read_events(logger: SessionLogger) -> list[dict]:
    return [json.loads(line) for line in
            logger.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


# ──────────────────────────────────────────────────── LOG_DIR (AC-18.4)
def test_log_dir_comes_from_the_environment(monkeypatch, tmp_path) -> None:
    """AC-18.4 — เดิม hardcode เป็น ROOT/logs แก้ไม่ได้เลย"""
    from vc import config as config_mod

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "somewhere"))
    assert config_mod.log_dir() == tmp_path / "somewhere"

    monkeypatch.setenv("LOG_DIR", "บันทึก")     # สัมพัทธ์ = อ้างจากรากโปรเจกต์
    assert config_mod.log_dir() == config_mod.ROOT / "บันทึก"

    monkeypatch.delenv("LOG_DIR")
    assert config_mod.log_dir() == config_mod.ROOT / "logs"


def test_logger_writes_into_the_configured_dir(tmp_path) -> None:
    log = SessionLogger(tmp_path / "vc-logs")
    try:
        assert log.dir.parent == tmp_path / "vc-logs"
        assert log.jsonl.exists()
        assert log.md.exists()
    finally:
        log.close()


# ────────────────────────────────────────────────── สิทธิ์ไฟล์ (AC-18.3)
@pytest.mark.skipif(WINDOWS, reason="สิทธิ์แบบ POSIX ใช้ไม่ได้บน Windows")
def test_new_log_dir_and_files_are_private(tmp_path) -> None:
    """AC-18.3 — dir 700 / ไฟล์ 600"""
    root = tmp_path / "logs"
    log = SessionLogger(root, save_audio=True)
    try:
        log.turn("user", "สวัสดี")
        log.save_utterance(np.zeros(160, np.int16), 16000, 1)
        for path in (root, log.dir, log.audio_dir):
            mode = stat.S_IMODE(path.stat().st_mode)
            assert mode == DIR_MODE, f"{path} = {oct(mode)} (ต้องเป็น 0o700)"
        files = [log.jsonl, log.md, log.audio_dir / "user-001.wav"]
        for path in files:
            mode = stat.S_IMODE(path.stat().st_mode)
            assert mode == FILE_MODE, f"{path} = {oct(mode)} (ต้องเป็น 0o600)"
    finally:
        log.close()


@pytest.mark.skipif(WINDOWS, reason="สิทธิ์แบบ POSIX ใช้ไม่ได้บน Windows")
def test_existing_wide_open_dir_is_tightened(tmp_path) -> None:
    root = tmp_path / "logs"
    root.mkdir()
    os.chmod(root, 0o777)
    log = SessionLogger(root)
    try:
        assert stat.S_IMODE(root.stat().st_mode) == DIR_MODE
    finally:
        log.close()


# ─────────────────────────────────────────── LOG_RETENTION_DAYS (AC-18.2/18.6)
def make_session_dir(root: Path, age_days: int) -> Path:
    stamp = (tz_now() - timedelta(days=age_days)).strftime("%Y%m%d-%H%M%S")
    path = root / f"session-{stamp}"
    path.mkdir(parents=True)
    (path / "transcript.md").write_text("เก่า", encoding="utf-8")
    return path


def test_old_sessions_are_removed_and_recent_ones_kept(tmp_path) -> None:
    """AC-18.2 — 10 วันหาย / 3 วันอยู่"""
    root = tmp_path / "logs"
    old, fresh = make_session_dir(root, 10), make_session_dir(root, 3)
    removed = purge_old_sessions(root, 7)
    assert removed == [old]
    assert not old.exists() and fresh.exists()


def test_retention_zero_keeps_everything(tmp_path) -> None:
    """AC-18.6 — 0 = ไม่ลบอัตโนมัติ ไม่ใช่ลบทุกอย่าง"""
    root = tmp_path / "logs"
    ancient = make_session_dir(root, 900)
    assert purge_old_sessions(root, 0) == []
    assert ancient.exists()


def test_purge_ignores_dirs_it_does_not_recognise(tmp_path) -> None:
    """ของที่ผู้ใช้เอามาวางเองต้องไม่ถูกลบทิ้ง"""
    root = tmp_path / "logs"
    root.mkdir()
    keep = root / "ของสำคัญ"
    keep.mkdir()
    (root / "session-ไม่ใช่วันที่").mkdir()
    (root / "README.txt").write_text("hi", encoding="utf-8")
    assert purge_old_sessions(root, 1) == []
    assert keep.exists() and (root / "session-ไม่ใช่วันที่").exists()


def test_logger_purges_on_startup_and_records_it(tmp_path) -> None:
    root = tmp_path / "logs"
    old = make_session_dir(root, 30)
    log = SessionLogger(root, retention_days=7)
    try:
        assert not old.exists()
        kinds = [e["type"] for e in read_events(log)]
        assert "logs_purged" in kinds
        assert log.dir.exists(), "โฟลเดอร์ของ session ปัจจุบันต้องไม่ถูกลบไปด้วย"
    finally:
        log.close()


def test_session_stamp_round_trip() -> None:
    """ชื่อโฟลเดอร์ไม่เก็บ tz ไว้ → ตีความในโซนที่ระบบใช้ แล้วติด tzinfo ให้ (P3-20)"""
    got = session_started_at(Path("session-20250101-101530"))
    assert got == datetime(2025, 1, 1, 10, 15, 30, tzinfo=ZoneInfo("Asia/Bangkok"))
    assert got.tzinfo is not None, "ต้องมี tzinfo ไม่งั้นเทียบกับ now() ไม่ได้"
    assert session_started_at(Path("อื่น ๆ")) is None


# ────────────────────────────────────────────────── LOG_TRANSCRIPT (AC-18.1)
def test_transcript_off_writes_no_words_anywhere(tmp_path) -> None:
    """AC-18.1 — ไม่มีไฟล์ transcript และคำพูดต้องไม่ไปโผล่ใน jsonl ด้วย"""
    log = SessionLogger(tmp_path / "logs", save_audio=True, transcript=False)
    secret = "เลขบัญชีของผมคือ 1234567890"
    try:
        log.turn("user", secret, epoch=1)
        log.event("asr", text=secret, latency_ms=120)
        assert log.save_utterance(np.zeros(160, np.int16), 16000, 1) is None
    finally:
        log.close()

    assert not log.md.exists(), "ต้องไม่มี transcript.md เลย"
    assert not log.audio_dir.exists(), "ปิด transcript แล้วต้องไม่เก็บไฟล์เสียงด้วย"
    raw = log.jsonl.read_text(encoding="utf-8")
    assert secret not in raw
    events = read_events(log)
    asr = [e for e in events if e["type"] == "asr"][0]
    assert asr["latency_ms"] == 120, "ตัวเลขวัดผลต้องยังอยู่"
    assert asr["text_chars"] == len(secret)
    assert [e for e in events if e["type"] == "session_end"], "ยังต้องปิด session ปกติ"


def test_transcript_on_is_unchanged(tmp_path) -> None:
    """ค่าเริ่มต้นต้องทำงานเหมือนก่อนแก้ทุกอย่าง"""
    log = SessionLogger(tmp_path / "logs", meta={"chat_model": "m"})
    try:
        log.turn("assistant", "สวัสดีครับ", epoch=1)
    finally:
        log.close()
    md = log.md.read_text(encoding="utf-8")
    assert "สวัสดีครับ" in md and "chat_model" in md
    assert "สวัสดีครับ" in log.jsonl.read_text(encoding="utf-8")


def test_config_reads_the_new_switches(monkeypatch) -> None:
    from vc.config import load_config

    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("LOG_TRANSCRIPT", "0")
    monkeypatch.setenv("LOG_RETENTION_DAYS", "14")
    cfg = load_config()
    assert cfg.log_transcript is False
    assert cfg.log_retention_days == 14


def test_defaults_keep_every_log(monkeypatch) -> None:
    from vc.config import load_config

    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("API_KEY", "k")
    for key in ("LOG_TRANSCRIPT", "LOG_RETENTION_DAYS", "LOG_DIR"):
        monkeypatch.delenv(key, raising=False)
    cfg = load_config()
    assert cfg.log_transcript is True
    assert cfg.log_retention_days == 0, "ค่าเริ่มต้นต้องไม่ลบบันทึกของผู้ใช้เอง"


# ────────────────────────────────────────── ห้ามส่ง log_dir ออกหน้าเว็บ (AC-18.5)
def test_ready_event_never_leaks_the_log_path(web_session, outbox) -> None:
    """AC-18.5 — regression guard ของ P1-1"""
    web_session.header()
    ready = outbox.of("ready")
    assert ready, "ต้องส่ง ready ออกไป"
    assert "log_dir" not in ready[0]
    blob = json.dumps(ready[0], ensure_ascii=False)
    assert "/Users/" not in blob and "logs" not in blob
