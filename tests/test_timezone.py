"""P3-20 — เวลาต้องเป็นเวลาไทยจริง ไม่ใช่เวลาของเครื่อง

`datetime.now()` เปล่าอ่านค่าจาก TZ ของ process บน container ที่ตั้ง TZ=UTC
(ค่าปริยายของ image ส่วนใหญ่) เวลาจึงเพี้ยนไป 7 ชั่วโมง ทั้งที่ system prompt
บอกโมเดลว่า "ตามเวลาประเทศไทย" — คำถามแนว "วันนี้/ตอนนี้" จึงตอบผิดวันได้
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from vc.config import DEFAULT_TZ, Config, now, thai_clock, zone
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


def test_unknown_zone_prompt_names_the_fallback_not_the_bad_value(capsys) -> None:
    bad_zone = "Invalid/Prompt_Zone"
    cfg = Config(tz=bad_zone, web_search=False)
    content = cfg.system_message()["content"]

    assert "ตามเวลาประเทศไทย" in content
    assert bad_zone not in content
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
# ระบุโฟลเดอร์ให้ชัด ห้าม ROOT.rglob(): บนเครื่องที่ทำตาม README แล้วมี .venv/
# อยู่ในโปรเจกต์ มันจะกวาด site-packages เข้ามาแล้วฟ้องโค้ดของ pydantic/pytest
# ส่วนบน path ที่มี `.claude` จะได้ลิสต์ว่างแล้วผ่านทั้งที่ไม่ได้ตรวจอะไรเลย
SOURCES = ([p for folder in ("web", "vc") for p in (ROOT / folder).rglob("*.py")]
           + [ROOT / "voice_chat.py"])


def naive_time_calls(source: str) -> list[tuple[int, str]]:
    """หาการอ่านเวลาแบบไม่ระบุ timezone จาก AST — ไม่ใช่จากข้อความดิบ

    ต้องดูที่ AST เพราะการค้นด้วย regex ทีละบรรทัดแยกไม่ออกว่าอะไรเป็นโค้ดจริง
    อะไรเป็นคอมเมนต์/docstring ที่ *เตือน* ว่าห้ามใช้ `datetime.now()` เปล่า
    (vc/config.py มีทั้งสองแบบ และเคยถูกฟ้องผิดมาแล้ว)
    """
    bad: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        no_args = not node.args and not node.keywords
        if name.endswith("datetime.now") and no_args:
            bad.append((node.lineno, f"{name}()"))
        elif name.endswith("utcnow"):
            bad.append((node.lineno, f"{name}()"))
        elif name.endswith("time.localtime") and no_args:
            bad.append((node.lineno, f"{name}()"))
    return sorted(bad)


def _dotted(node: ast.AST) -> str:
    """ชื่อแบบจุดของสิ่งที่ถูกเรียก เช่น `datetime.now`, `time.localtime`"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_the_source_list_is_not_empty() -> None:
    """กันเทสต์ข้างล่างผ่านเพราะไม่มีไฟล์ให้ตรวจ"""
    assert SOURCES, "ไม่พบไฟล์ต้นฉบับให้ตรวจเลย"
    assert (ROOT / "vc" / "config.py") in SOURCES


def test_the_checker_catches_the_real_thing() -> None:
    """ตัวตรวจต้อง trigger จริง ไม่ใช่แค่คืนลิสต์ว่างเสมอ"""
    assert naive_time_calls("from datetime import datetime\nx = datetime.now()\n")
    assert naive_time_calls("import datetime\nx = datetime.datetime.utcnow()\n")
    assert naive_time_calls("import time\nx = time.localtime()\n")


def test_the_checker_ignores_prose_that_merely_mentions_it() -> None:
    """docstring/คอมเมนต์ที่ห้ามใช้ `datetime.now()` ไม่ใช่การละเมิด"""
    assert naive_time_calls('"""ห้ามใช้ `datetime.now()` เปล่า"""\n') == []
    assert naive_time_calls("# datetime.now() เปล่าให้เวลาเพี้ยน\n") == []
    assert naive_time_calls('BAD = "datetime.utcnow()"\n') == []
    # ของที่ถูกต้องก็ต้องไม่ถูกฟ้อง
    assert naive_time_calls("datetime.now(zone(tz))\ntime.localtime(0)\n") == []


def test_no_naive_datetime_now_anywhere_in_the_source() -> None:
    """AC-20.2 — `datetime.now()` เปล่าไม่มีสิทธิ์อยู่ในโปรเจกต์นี้อีก"""
    assert SOURCES
    bad = [f"{path.relative_to(ROOT)}:{line}: {call}"
           for path in SOURCES
           for line, call in naive_time_calls(path.read_text(encoding="utf-8"))]
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


def test_windows_declares_the_timezone_database_dependency() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(
        r"(?mi)^tzdata[^\n;]*;[^\n]*sys_platform\s*==\s*[\"']win32[\"']",
        requirements,
    ), "Windows ไม่มี IANA timezone database จึงต้องติดตั้ง tzdata โดยตรง"


# ─────────────────────────── เวลาที่ยื่นให้โมเดลต้องเป็นคำที่คนไทยพูดจริง
# ส่งแค่ "เวลา 22:39 น." แล้วให้โมเดลแปลงเอง ได้ "22 โมงกว่าๆ" ซึ่งไม่มีใครพูด
# และมันถูกอ่านออกเสียงใส่หูผู้ใช้ตรง ๆ
@pytest.mark.parametrize("hour,minute,said", [
    (0, 0, "เที่ยงคืน"),
    (1, 0, "ตีหนึ่ง"),
    (5, 30, "ตีห้าครึ่ง"),
    (6, 0, "หกโมงเช้า"),
    (11, 0, "สิบเอ็ดโมงเช้า"),
    (12, 0, "เที่ยงวัน"),
    (13, 0, "บ่ายโมง"),
    (15, 0, "บ่ายสามโมง"),
    (16, 0, "สี่โมงเย็น"),
    (18, 0, "หกโมงเย็น"),
    (19, 0, "หนึ่งทุ่ม"),
    (22, 39, "สี่ทุ่มสามสิบเก้านาที"),
    (23, 59, "ห้าทุ่มห้าสิบเก้านาที"),
])
def test_the_clock_is_read_the_way_thai_speakers_say_it(hour, minute, said) -> None:
    assert thai_clock(datetime(2026, 8, 20, hour, minute)) == said


def test_no_hour_is_ever_read_as_a_number_of_moong() -> None:
    """ภาษาไทยไม่มี "โมง" เกินสิบเอ็ด — ทุกชั่วโมงต้องมีคำเรียกของตัวเอง"""
    for hour in range(24):
        said = thai_clock(datetime(2026, 8, 20, hour, 0))
        assert said and not any(ch.isdigit() for ch in said)


def test_the_system_prompt_hands_the_model_the_thai_words(cfg) -> None:
    content = cfg.system_message()["content"]
    assert thai_clock(cfg.now()) in content, "โมเดลต้องได้คำอ่าน ไม่ใช่แค่ตัวเลข"
