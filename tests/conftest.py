"""ค่าตั้งร่วมของชุดทดสอบ pytest

หน้าที่หลักสามอย่าง
1. ให้ import โมดูลของโปรเจกต์ได้โดยไม่ต้องติดตั้งเป็นแพ็กเกจ
2. ปลอม ``sounddevice`` ให้เมื่อเครื่องไม่มี PortAudio (เครื่อง CI/เซิร์ฟเวอร์)
   เทสต์จึงรันได้แม้ไม่มีไมค์/ลำโพง — ดู AC-0.2
3. ข้ามสคริปต์ทดสอบรุ่นเดิม 5 ไฟล์ (ยังรันตรง ๆ ด้วย ``python3 tests/test_xxx.py``
   ได้เหมือนเดิม) เพราะเขียนแบบ ``main()`` ไม่ใช่รูปแบบ pytest และบางไฟล์
   ต้องต่อเน็ตจริง/เปิดเบราว์เซอร์ — การย้ายเข้ามาเป็น pytest อยู่ในงาน P3-22
"""
from __future__ import annotations

import queue
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────── เครื่องที่ไม่มี PortAudio (AC-0.2)
def _install_sounddevice_stub() -> None:
    """ยัดโมดูล sounddevice ปลอมเข้า sys.modules ถ้าของจริงใช้ไม่ได้

    ``vc.audio`` import sounddevice ตั้งแต่ระดับโมดูล ถ้าเครื่องไม่มี PortAudio
    จะ OSError ตั้งแต่ import ทำให้เก็บเทสต์ไม่ได้เลย (ไม่ใช่ผลลัพธ์ที่ต้องการ)
    """
    try:
        import sounddevice  # noqa: F401
        return
    except Exception:        # noqa: BLE001 — ไม่มีไลบรารี/ไม่มี PortAudio/ไม่มีอุปกรณ์
        pass

    stub = types.ModuleType("sounddevice")

    class _Stream:                                    # pragma: no cover - ตัวปลอม
        def __init__(self, *a, **k) -> None:
            self.samplerate = k.get("samplerate", 16000)
            self.channels = k.get("channels", 1)

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def close(self) -> None: ...
        def write(self, *a, **k) -> None: ...
        def read(self, n):
            return np.zeros((n, 1), np.int16), False

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None: ...

    stub.InputStream = _Stream                        # type: ignore[attr-defined]
    stub.OutputStream = _Stream                       # type: ignore[attr-defined]
    stub.RawInputStream = _Stream                     # type: ignore[attr-defined]
    stub.RawOutputStream = _Stream                    # type: ignore[attr-defined]
    stub.PortAudioError = OSError                     # type: ignore[attr-defined]
    stub.query_devices = lambda *a, **k: []           # type: ignore[attr-defined]
    stub.default = types.SimpleNamespace(             # type: ignore[attr-defined]
        device=(None, None), samplerate=None, channels=(1, 1))
    stub.__stub__ = True                              # type: ignore[attr-defined]
    sys.modules["sounddevice"] = stub


_install_sounddevice_stub()


# ─────────────────────────────────────── สคริปต์ทดสอบรุ่นเดิม (ยังใช้ได้ตามเดิม)
LEGACY_SCRIPTS = [
    "test_offline.py",     # รันเอง: python3 tests/test_offline.py
    "test_web.py",
    "test_search.py",      # ส่วน [4]-[5] ต่อเน็ตจริง
    "test_web_e2e.py",     # ต้องเรียก API จริง
    "test_browser.py",     # ต้องมี Playwright + Chromium
]
collect_ignore = list(LEGACY_SCRIPTS)


# ────────────────────────────────────────────────────────────── ของใช้ร่วมกัน
class FakeOutbox:
    """Outbox ปลอม — เก็บสิ่งที่ session ส่งออกไว้ในลิสต์แทนการยิงเข้า WebSocket"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.closed = False

    def json(self, type: str, **fields) -> None:      # noqa: A002 - ตามสัญญาเดิม
        fields["type"] = type
        self.sent.append(("json", fields))

    def binary(self, data: bytes) -> None:
        self.sent.append(("bin", data))

    def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------ ตัวช่วยอ่านผล
    def of(self, kind: str) -> list[dict]:
        return [p for k, p in self.sent
                if k == "json" and isinstance(p, dict) and p.get("type") == kind]

    def kinds(self) -> list[str]:
        return [p.get("type") for k, p in self.sent
                if k == "json" and isinstance(p, dict)]


@pytest.fixture
def outbox() -> FakeOutbox:
    return FakeOutbox()


@pytest.fixture
def cfg(tmp_path):
    """คอนฟิกสำหรับเทสต์ — ไม่แตะ .env ของเครื่องและไม่เขียนลง logs/ ของจริง"""
    from vc.config import Config

    return Config(
        base_url="http://127.0.0.1:9",     # พอร์ต discard — กันเผลอยิงออกจริง
        api_key="test-key",
        chat_model="fake-chat",
        asr_model="fake-asr",
        tts_model="fake-tts",
        log_dir=tmp_path / "logs",
        web_search=False,
    )


@pytest.fixture
def web_session(cfg, outbox):
    """WebSession จริง แต่เชื่อมกับ Outbox ปลอมและไม่มีการเรียก API"""
    from web.session import WebSession

    session = WebSession(cfg, outbox)      # type: ignore[arg-type]
    session.api.transcribe = lambda pcm, sr: ""        # type: ignore[method-assign]
    session.api.synthesize = lambda text, sr: np.zeros(240, np.int16)  # type: ignore[method-assign]
    try:
        yield session
    finally:
        session.running.clear()
        session.cancel.set()
        _drain(session.tts_queue)
        try:
            session.api.close()
        except Exception:      # noqa: BLE001
            pass


def _drain(q: "queue.Queue") -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
