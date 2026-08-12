"""P3-20 — เวลาต้องเป็นเวลาไทยจริง ไม่ใช่เวลาของเครื่อง

`datetime.now()` เปล่าอ่านค่าจาก TZ ของ process บน container ที่ตั้ง TZ=UTC
(ค่าปริยายของ image ส่วนใหญ่) เวลาจึงเพี้ยนไป 7 ชั่วโมง ทั้งที่ system prompt
บอกโมเดลว่า "ตามเวลาประเทศไทย" — คำถามแนว "วันนี้/ตอนนี้" จึงตอบผิดวันได้
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from vc.config import DEFAULT_TZ, Config, now, zone
from vc.logger import SessionLogger

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def test_default_zone_is_bangkok(monkeypatch) -> None:
    monkeypatch.delenv("APP_TZ", raising=False)
    assert zone() == ZoneInfo("Asia/Bangkok")
    assert now().tzinfo is not None


def test_app_tz_overrides_it(monkeypatch) -> None:
    """AC-20.3"""
    monkeypatch.setenv("APP_TZ", "UTC")
    assert zone() == ZoneInfo("UTC")
    assert now().utcoffset().total_seconds() == 0


def test_unknown_zone_falls_back_with_a_warning(monkeypatch, capsys) -> None:
    monkeypatch.setenv("APP_TZ", "Mars/Olympus_Mons")
    assert zone() == ZoneInfo(DEFAULT_TZ)
    assert "APP_TZ" in capsys.readouterr().err


def test_machine_tz_does_not_change_the_answer(monkeypatch) -> None:
    """AC-20.1 — ตั้ง TZ=UTC ให้ process แล้วเวลาที่ระบบใช้ต้องยังเป็นเวลาไทย"""
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.delenv("APP_TZ", raising=False)
    reference = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Bangkok"))
    got = now()
    assert got.utcoffset().total_seconds() == 7 * 3600
    assert abs((got - reference).total_seconds()) < 5


def test_system_prompt_says_the_thai_time(monkeypatch) -> None:
    monkeypatch.delenv("APP_TZ", raising=False)
    cfg = Config(web_search=False)
    content = cfg.system_message()["content"]
    stamp = now(cfg.tz)
    assert f"เวลา {stamp:%H:%M} น." in content
    assert "ตามเวลาประเทศไทย" in content


def test_system_prompt_states_the_real_zone_when_it_is_not_thai() -> None:
    """ห้ามบอกโมเดลว่า 'เวลาประเทศไทย' ถ้าตั้ง APP_TZ เป็นโซนอื่น"""
    cfg = Config(tz="UTC", web_search=False)
    content = cfg.system_message()["content"]
    assert "ตามเขตเวลา UTC" in content
    assert "ตามเวลาประเทศไทย" not in content


def test_session_dir_name_and_timestamps_share_one_zone(tmp_path) -> None:
    """AC-20.4 — ชื่อโฟลเดอร์กับ ts ในไฟล์ต้องเป็นโซนเดียวกัน"""
    log = SessionLogger(tmp_path / "logs", tz="UTC")
    try:
        log.turn("user", "ทดสอบ")
    finally:
        log.close()
    stamp = log.dir.name.removeprefix("session-")
    first = log.jsonl.read_text(encoding="utf-8").splitlines()[0]
    ts = re.search(r'"ts": "([^"]+)"', first).group(1)
    assert ts.endswith("+00:00"), f"ts ต้องบอก offset ของ UTC: {ts}"
    assert ts[:10].replace("-", "") == stamp[:8], "วันที่ในชื่อโฟลเดอร์กับใน ts ไม่ตรงกัน"


def test_logger_uses_the_config_zone(tmp_path) -> None:
    log = SessionLogger(tmp_path / "logs", tz="Asia/Bangkok")
    try:
        assert log.started.tzinfo == ZoneInfo("Asia/Bangkok")
    finally:
        log.close()


# ────────────────────────────────────────────────── กันการถอยหลัง (AC-20.2)
SOURCES = [p for p in ROOT.rglob("*.py")
           if ".claude" not in p.parts and "tests" not in p.parts]


def test_no_naive_datetime_now_anywhere_in_the_source() -> None:
    """AC-20.2 — `datetime.now()` เปล่าไม่มีสิทธิ์อยู่ในโปรเจกต์นี้อีก"""
    bad = []
    for path in SOURCES:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"datetime\.now\(\s*\)|datetime\.utcnow\(|"
                         r"time\.localtime\(\s*\)", line):
                bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert bad == [], "พบการอ่านเวลาแบบไม่ระบุ timezone:\n" + "\n".join(bad)


def test_startup_under_utc_env_still_reports_thai_time() -> None:
    """รันจริงในกระบวนการที่ TZ=UTC (ไม่ใช่แค่ monkeypatch)"""
    code = (
        "from vc.config import Config, now;"
        "c = Config(web_search=False);"
        "print(now(c.tz).strftime('%z'), c.system_message()['content'][:0])"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                         capture_output=True, text=True,
                         env={"TZ": "UTC", "PATH": "/usr/bin:/bin"})
    assert out.stdout.strip() == "+0700"
