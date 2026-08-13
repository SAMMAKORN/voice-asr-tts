"""P2-6 — ApiClient เป็นขอบเขตจัดการ error จุดเดียว + ลองใหม่แบบ backoff

ทุกเทสต์ใช้ MockTransport ของ httpx จึงไม่มี request ออกเน็ตจริงแม้แต่ครั้งเดียว
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from vc.api import ApiClient, ApiError, pcm16_to_wav

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
PCM = np.zeros(1600, np.int16)


def client(cfg, handler, *, retry_max: int = 2, base_ms: int = 300) -> ApiClient:
    """ApiClient ที่ทุก HTTP call วิ่งผ่าน handler ที่เราคุมเอง"""
    cfg.http_retry_max = retry_max
    cfg.http_retry_base_ms = base_ms
    api = ApiClient(cfg)
    transport = httpx.MockTransport(handler)
    for name in ("_audio", "_chat"):
        old: httpx.Client = getattr(api, name)
        setattr(api, name, httpx.Client(base_url=old.base_url, transport=transport,
                                        timeout=old.timeout))
        old.close()
    return api


def wav_response() -> httpx.Response:
    return httpx.Response(200, content=pcm16_to_wav(PCM, 16000),
                          headers={"content-type": "audio/wav"})


def sse(*chunks: str) -> bytes:
    body = "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def token(text: str) -> str:
    return '{"choices":[{"delta":{"content":"%s"}}]}' % text


# ───────────────────────────────────────────────── ไม่มี httpx หลุดออกนอก vc/api
def test_connect_error_becomes_api_error(cfg) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("เชื่อมต่อไม่ได้", request=request)

    api = client(cfg, handler, base_ms=1)
    try:
        with pytest.raises(ApiError):
            api.transcribe(PCM, 16000)
    finally:
        api.close()
    assert len(calls) == 3, f"ต้องลองใหม่ครบ 2 ครั้ง (ยิงจริง {len(calls)})"


def test_read_timeout_becomes_api_error(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("ช้าเกิน", request=request)

    api = client(cfg, handler, retry_max=0)
    try:
        with pytest.raises(ApiError):
            api.synthesize("สวัสดี", 24000)
    finally:
        api.close()


def test_only_api_module_touches_httpx() -> None:
    """AC-6.5 — httpx ต้องเหลือใช้เฉพาะ vc/api.py และ vc/websearch.py"""
    hits: list[str] = []
    for path in [ROOT / "voice_chat.py", *(ROOT / "web").glob("*.py"),
                 *(ROOT / "vc").glob("*.py")]:
        if path.name in ("api.py", "websearch.py") and path.parent.name == "vc":
            continue
        if re.search(r"^\s*(import httpx|from httpx)", path.read_text("utf-8"),
                     re.MULTILINE):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], f"ไฟล์เหล่านี้ยังใช้ httpx ตรง ๆ: {hits}"


# ─────────────────────────────────────────────────────────── retry ตามเงื่อนไข
def test_retries_503_then_succeeds_with_backoff(cfg) -> None:
    """AC-6.2 — 503 สองครั้งแล้วสำเร็จ ผู้ใช้ไม่เห็น error และหน่วงตาม backoff"""
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        if len(calls) < 3:
            return httpx.Response(503, text="ไม่พร้อมให้บริการ")
        return httpx.Response(200, json={"text": "ได้ยินแล้วครับ"})

    api = client(cfg, handler)
    t0 = time.monotonic()
    try:
        assert api.transcribe(PCM, 16000) == "ได้ยินแล้วครับ"
    finally:
        api.close()
    elapsed = time.monotonic() - t0
    assert len(calls) == 3
    # 300ms + 600ms ± jitter 20%
    assert 0.6 <= elapsed <= 1.6, f"หน่วงไม่สอดคล้องกับ backoff ({elapsed:.2f}s)"


def test_no_retry_on_401(cfg) -> None:
    """AC-6.3 — 4xx ที่ไม่ใช่ 429 ต้องไม่ลองใหม่เลย"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="token ผิด")

    api = client(cfg, handler)
    try:
        with pytest.raises(ApiError) as err:
            api.transcribe(PCM, 16000)
    finally:
        api.close()
    assert len(calls) == 1, f"ยิงซ้ำทั้งที่เป็น 401 ({len(calls)} ครั้ง)"
    assert "401" in str(err.value)


def test_retry_after_header_is_respected(cfg) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0.2"}, text="ช้าลง")
        return wav_response()

    api = client(cfg, handler, base_ms=5000)   # base ใหญ่มาก ถ้าไม่เคารพ header จะช้า
    t0 = time.monotonic()
    try:
        assert api.synthesize("สวัสดี", 16000).size == PCM.size
    finally:
        api.close()
    elapsed = time.monotonic() - t0
    assert len(calls) == 2
    assert elapsed < 1.0, f"ไม่เคารพ Retry-After (รอ {elapsed:.2f}s)"


# ──────────────────────────────────────────────────────── สตรีมของ chat
class _Broken(httpx.SyncByteStream):
    """สตรีมที่ส่ง token ออกไปแล้วขาดกลางทาง"""

    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix

    def __iter__(self):
        yield self.prefix
        raise httpx.ReadError("สตรีมขาด")

    def close(self) -> None:
        pass


def test_chat_stream_retries_before_first_token(cfg) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="ยังไม่พร้อม")
        return httpx.Response(200, content=sse(token("ได้ครับ")))

    api = client(cfg, handler, base_ms=1)
    try:
        out = "".join(api.chat_stream([{"role": "user", "content": "ถาม"}],
                                      threading.Event()))
    finally:
        api.close()
    assert out == "ได้ครับ"
    assert len(calls) == 2


def test_chat_stream_never_retries_after_first_token(cfg) -> None:
    """AC-6.4 — หลุดหลังพูดไปแล้ว ห้ามยิงซ้ำ (ไม่งั้นผู้ใช้ได้ยินท่อนเดิมสองครั้ง)"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, stream=_Broken(b"data: " + token("ครึ่งแรก").encode()
                                                  + b"\n\n"),
                              headers={"content-type": "text/event-stream"})

    api = client(cfg, handler, base_ms=1)
    got: list[str] = []
    try:
        with pytest.raises(ApiError):
            for delta in api.chat_stream([{"role": "user", "content": "ถาม"}],
                                         threading.Event()):
                got.append(delta)
    finally:
        api.close()
    assert got == ["ครึ่งแรก"]
    assert len(calls) == 1, f"ยิงซ้ำหลังมี token ออกไปแล้ว ({len(calls)} ครั้ง)"


def test_chat_stream_http_error_becomes_api_error(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="คำขอผิดรูป")

    api = client(cfg, handler, base_ms=1)
    try:
        with pytest.raises(ApiError) as err:
            list(api.chat_stream([{"role": "user", "content": "x"}],
                                 threading.Event()))
    finally:
        api.close()
    assert "400" in str(err.value)
