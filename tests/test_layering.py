"""P3-21 — ชั้นของโค้ดต้องไม่กลับด้าน

เดิม `web/session.py` import จาก `voice_chat.py` (สคริปต์ CLI) ซึ่ง import
`sounddevice` ที่ระดับโมดูล → เว็บเซิร์ฟเวอร์รันบนคอนเทนเนอร์ที่ไม่มี PortAudio
ไม่ได้เลย ทั้งที่โหมดเว็บใช้ไมค์/ลำโพงของเบราว์เซอร์
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent

# บล็อกการ import sounddevice ให้เหมือนเครื่องที่ไม่มี PortAudio ติดตั้ง
BLOCK_SOUNDDEVICE = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise ImportError("จำลองเครื่องที่ไม่มี PortAudio")
        return None

sys.meta_path.insert(0, Blocker())
"""


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK_SOUNDDEVICE + code],
                          cwd=ROOT, capture_output=True, text=True)


# ─────────────────────────────────────────── รันได้บนเครื่องที่ไม่มีอุปกรณ์เสียง (AC-21.1)
def test_sounddevice_is_really_blocked_in_the_subprocess() -> None:
    """กันเทสต์ข้างล่างผ่านเพราะการบล็อกไม่ทำงาน"""
    out = run("import sounddevice")
    assert out.returncode != 0 and "PortAudio" in out.stderr


def test_web_stack_imports_without_portaudio() -> None:
    """AC-21.1"""
    out = run("import web.session, web.server, vc.chat; print('ok')")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_a_web_session_can_be_built_without_portaudio() -> None:
    """AC-21.1 — ไม่ใช่แค่ import ได้ แต่ต้องสร้าง session และส่ง ready ได้จริง"""
    out = run("""
import tempfile, pathlib
from vc.config import Config
from web.session import WebSession

class Out:
    def __init__(self): self.sent = []
    def json(self, type, **f): self.sent.append(dict(f, type=type))
    def binary(self, data): ...
    def close(self): ...

with tempfile.TemporaryDirectory() as tmp:
    cfg = Config(base_url="http://127.0.0.1:9", api_key="k",
                 log_dir=pathlib.Path(tmp), web_search=False)
    s = WebSession(cfg, Out())
    s.header()
    print([m["type"] for m in s.out.sent])
    s.request_stop()
    s.close_resources()
""")
    assert out.returncode == 0, out.stderr
    assert "ready" in out.stdout


def test_terminal_mode_explains_what_is_missing() -> None:
    """ฝั่ง CLI ต้องได้ข้อความที่บอกวิธีแก้ ไม่ใช่ ImportError ดิบ"""
    out = run("""
from vc.audio import Microphone
try:
    Microphone(16000, 20).start()
except RuntimeError as exc:
    print("RuntimeError:", "web.server" in str(exc), "portaudio" in str(exc).lower())
""")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "RuntimeError: True True"


def test_audio_module_has_no_module_level_import() -> None:
    """AC-21.1 — ต้อง import ข้างในฟังก์ชันที่ใช้จริงเท่านั้น"""
    src = (ROOT / "vc" / "audio.py").read_text(encoding="utf-8")
    module_level = [line for line in src.splitlines()
                    if re.match(r"^(import|from) sounddevice", line)]
    assert module_level == [], f"ยัง import ที่ระดับโมดูล: {module_level}"
    assert "def sd()" in src
    inside = [line for line in src.splitlines()
              if re.match(r"^\s+import sounddevice", line)]
    assert inside, "ต้องมี import อยู่ข้างในฟังก์ชัน"


# ────────────────────────────────────────────────── ทิศทางการ import (AC-21.2/21.3)
def test_core_and_web_never_import_the_cli_script() -> None:
    """AC-21.2 — `grep -rn "import voice_chat|from voice_chat" web/ vc/` = 0"""
    hits = [f"{p.relative_to(ROOT)}:{i}"
            for folder in ("web", "vc")
            for p in (ROOT / folder).rglob("*.py")
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"^\s*(import voice_chat|from voice_chat)", line)]
    assert hits == [], f"ชั้นล่างยัง import สคริปต์ CLI: {hits}"


def test_core_knows_nothing_about_the_argument_parser() -> None:
    """AC-21.3 — `grep -rn "argparse" vc/` = 0"""
    hits = [f"{p.relative_to(ROOT)}:{i}"
            for p in (ROOT / "vc").rglob("*.py")
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if "argparse" in line]
    assert hits == [], f"แกนกลางยังอ้างถึงตัวแยก argument: {hits}"


def test_the_core_class_lives_in_the_core_package() -> None:
    import vc.chat

    from voice_chat import VoiceChat

    assert VoiceChat.__module__ == "vc.chat"
    assert vc.chat.VoiceChat is VoiceChat


def test_old_import_paths_still_work() -> None:
    """ของเดิมที่ import จาก voice_chat ต้องใช้ได้เหมือนเดิม (ไม่ทำให้ใครพัง)"""
    import voice_chat

    for name in ("VoiceChat", "GREETING", "SEARCH_FILLER", "CUT_MARK",
                 "NO_ANSWER_MARK", "FINDINGS_HEADER", "EVENTS_MAXSIZE",
                 "TTS_WAIT_TIMEOUT", "TTS_WAIT_POLL", "LOW_INPUT_VOLUME",
                 "phase_log", "mic_check", "selftest", "main"):
        assert hasattr(voice_chat, name), name


# ─────────────────────────────────────────────────────── RuntimeOptions (งานใหม่)
def test_runtime_options_defaults_match_the_old_namespace() -> None:
    from vc.options import RuntimeOptions

    opts = RuntimeOptions()
    assert opts.no_mic is False and opts.greet is True


def test_voice_chat_can_be_built_without_options(cfg) -> None:
    """ไม่ส่ง options มา = ค่าปริยาย (เดิมพัง เพราะต้องมี Namespace เสมอ)"""
    from vc.chat import VoiceChat
    from vc.options import RuntimeOptions

    chat = VoiceChat(cfg)
    try:
        assert isinstance(chat.args, RuntimeOptions)
        assert chat.args.greet is True
    finally:
        chat.close_resources()


def test_web_session_uses_runtime_options(web_session) -> None:
    from vc.options import RuntimeOptions

    assert isinstance(web_session.args, RuntimeOptions)
    assert web_session.args.no_mic is False and web_session.args.greet is True


# ────────────────────────────────────────────────────── selftest ชุดเดียว (AC-21.5)
def test_both_front_ends_call_the_same_selftest() -> None:
    """AC-21.5 — เดิมเขียนซ้ำสองชุดและพฤติกรรมต่างกันไปแล้ว"""
    import vc.selftest
    import voice_chat
    import web.session

    assert web.session.run_selftest is vc.selftest.run_selftest
    assert voice_chat.run_selftest is vc.selftest.run_selftest
    src = (ROOT / "web" / "session.py").read_text(encoding="utf-8")
    assert "def run_selftest" not in src, "ยังมีสำเนาที่สองอยู่ใน web/session.py"


def test_cli_selftest_only_renders(monkeypatch, cfg, capsys) -> None:
    """AC-21.5 — CLI ต้องไม่มีตรรกะการตรวจของตัวเอง แค่วาดผลที่ได้มา"""
    import voice_chat

    result = {"ok": True, "steps": [
        {"name": "TTS", "model": "m1", "ok": True, "detail": "ได้เสียง", "ms": 120},
        {"name": "ASR", "model": "m2", "ok": True, "detail": "ถอดได้", "ms": 80},
    ], "greeting": "สวัสดี"}
    monkeypatch.setattr(voice_chat, "run_selftest", lambda c: result)
    assert voice_chat.selftest(cfg) == 0
    out = capsys.readouterr().out
    assert "TTS" in out and "ได้เสียง" in out and "พร้อมใช้งาน" in out


def test_cli_selftest_reports_failure(monkeypatch, cfg) -> None:
    import voice_chat

    monkeypatch.setattr(voice_chat, "run_selftest", lambda c: {
        "ok": False, "greeting": "", "steps": [
            {"name": "ผิดพลาด", "model": "", "ok": False, "detail": "เชื่อมต่อไม่ได้",
             "ms": 0}]})
    assert voice_chat.selftest(cfg) == 1


def test_selftest_result_shape(monkeypatch, cfg) -> None:
    """โครงสร้างที่ทั้งสองฝ่ายพึ่งพา — เปลี่ยนแล้วหน้าเว็บพังเงียบ"""
    import numpy as np

    from vc import selftest as mod

    monkeypatch.setattr("vc.api.ApiClient.synthesize",
                        lambda self, text, sr: np.zeros(sr, np.int16))
    monkeypatch.setattr("vc.api.ApiClient.transcribe",
                        lambda self, pcm, sr: "ถอดเสียงได้")
    monkeypatch.setattr("vc.api.ApiClient.chat_stream",
                        lambda self, msgs, cancel, **kw: iter(["สวัสดี", "ครับ"]))
    result = mod.run_selftest(cfg)
    assert result["ok"] is True
    assert [s["name"] for s in result["steps"]] == ["TTS", "ASR", "LLM"]
    assert result["greeting"]
    for step in result["steps"]:
        assert set(step) == {"name", "model", "ok", "detail", "ms"}


def test_selftest_stops_reading_a_runaway_reply(monkeypatch, cfg) -> None:
    """พฤติกรรมที่รวมมาจากฝั่งเว็บ: ไม่อ่านคำตอบยาวไม่จำกัดตอนตรวจระบบ"""
    import numpy as np

    from vc import selftest as mod

    seen = {"n": 0}

    def stream(self, msgs, cancel, **kw):
        for _ in range(1000):
            if cancel.is_set():
                return
            seen["n"] += 1
            yield "ก" * 50

    monkeypatch.setattr("vc.api.ApiClient.synthesize",
                        lambda self, text, sr: np.zeros(sr, np.int16))
    monkeypatch.setattr("vc.api.ApiClient.transcribe", lambda self, pcm, sr: "x")
    monkeypatch.setattr("vc.api.ApiClient.chat_stream", stream)
    mod.run_selftest(cfg)
    assert seen["n"] < 20, f"อ่านไปถึง {seen['n']} ก้อน (ควรหยุดที่ ~400 ตัวอักษร)"


# ─────────────────────────────────────────────────── CLI เดิมต้องไม่เปลี่ยน (AC-21.6)
OLD_FLAGS = ["--list-devices", "--selftest", "--mic-check", "--input-device",
             "--output-device", "--model", "--headphones", "--no-tts", "--no-mic",
             "--no-web", "--mic-gain", "--one-voice", "--stream-tts", "--no-greet",
             "--save-audio", "--threshold"]


def test_cli_keeps_every_flag_it_had() -> None:
    """AC-21.6"""
    out = subprocess.run([sys.executable, "voice_chat.py", "--help"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    missing = [f for f in OLD_FLAGS if f not in out.stdout]
    assert missing == [], f"argument ที่หายไป: {missing}"


def test_cli_is_thin() -> None:
    """ไฟล์ CLI ต้องเหลือแค่การต่อสาย ไม่ใช่ที่อยู่ของตรรกะหลักอีก"""
    src = (ROOT / "voice_chat.py").read_text(encoding="utf-8")
    assert "class VoiceChat" not in src
    assert len(src.splitlines()) < 300, "CLI บวมกลับมาอีกแล้ว"
