"""P1-3 — เปิดหน้าเว็บต้องมีขอบเขต: redirect, ขนาด, ชนิดเนื้อหา, พอร์ต, เวลา

ใช้ MockTransport ของ httpx ทั้งหมด จึงไม่มี request ออกไปจริงแม้แต่ครั้งเดียว
"""
from __future__ import annotations

import time

import httpx
import pytest

from vc import websearch

pytestmark = pytest.mark.unit

HTML = {"content-type": "text/html; charset=utf-8"}


def transport(handler):
    return httpx.MockTransport(handler)


def page(text: str = "<title>หัวข้อ</title><p>เนื้อหา</p>") -> httpx.Response:
    return httpx.Response(200, headers=HTML, text=text)


# ───────────────────────────────────────────────────────────────────── AC-3.1
def test_redirect_into_private_network_is_blocked(monkeypatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/"})
        return page()          # ไม่ควรมาถึงตรงนี้เลย

    monkeypatch.setattr(websearch, "_is_public",
                        lambda host: host == "public.example")

    with pytest.raises(websearch.SearchError) as err:
        websearch.fetch_page("http://public.example/go", transport=transport(handler))

    assert "เครือข่ายภายใน" in str(err.value)
    assert seen == ["http://public.example/go"], (
        f"มี request ออกไปหาปลายทางที่ไม่ควรต่อ: {seen}")


def test_relative_redirect_is_resolved_and_followed(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/go":
            return httpx.Response(302, headers={"location": "/real"})
        return page("<title>ถึงแล้ว</title><p>ข้อความ</p>")

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    out = websearch.fetch_page("http://public.example/go", transport=transport(handler))
    assert "ถึงแล้ว" in out and "http://public.example/real" in out


# ───────────────────────────────────────────────────────────────────── AC-3.2
def test_too_many_redirects(monkeypatch) -> None:
    hops: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(request.url.path)
        n = int(request.url.path.strip("/h") or 0)
        return httpx.Response(302, headers={"location": f"/h{n + 1}"})

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    with pytest.raises(websearch.SearchError) as err:
        websearch.fetch_page("http://public.example/h0", transport=transport(handler))

    assert "too many redirects" in str(err.value)
    assert len(hops) == websearch.MAX_HOPS + 1, hops


# ───────────────────────────────────────────────────────────────────── AC-3.3
def test_body_is_capped(monkeypatch) -> None:
    served = {"bytes": 0}

    def big():
        for _ in range(50 * 1024 // 16):        # ~50 MB ถ้าอ่านจนจบ
            served["bytes"] += 16_384
            yield b"<p>" + b"x" * 16_378 + b"</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML, content=big())

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    out = websearch.fetch_page("http://public.example/big", max_chars=10 ** 9,
                               max_bytes=64_000, transport=transport(handler))

    assert served["bytes"] <= 64_000 + 16_384, (
        f"อ่านเข้ามา {served['bytes']} ไบต์ทั้งที่เพดาน 64000")
    assert len(out.encode()) < 200_000


def test_compressed_body_is_rejected_before_decompression(monkeypatch) -> None:
    """gzip bomb ต้องไม่ถูกคลายทั้งก้อนก่อนตรวจ FETCH_MAX_BYTES"""
    read = {"bytes": 0}
    accept_encodings: list[str] = []

    def compressed_payload():
        read["bytes"] += 1024
        yield b"not-really-gzip" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        accept_encodings.append(request.headers.get("accept-encoding", ""))
        return httpx.Response(
            200,
            headers={**HTML, "content-encoding": "gzip"},
            content=compressed_payload(),
        )

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    with pytest.raises(websearch.SearchError) as err:
        websearch.fetch_page(
            "http://public.example/bomb", max_bytes=10_000,
            transport=transport(handler))

    assert "บีบอัด" in str(err.value)
    assert read["bytes"] == 0, "อ่าน body ก่อนปฏิเสธ Content-Encoding"
    assert accept_encodings == ["identity"]


def test_fetch_max_bytes_env_is_used(monkeypatch) -> None:
    monkeypatch.setenv("FETCH_MAX_BYTES", "2048")
    assert websearch._max_bytes() == 2048
    monkeypatch.setenv("FETCH_MAX_BYTES", "ไม่ใช่ตัวเลข")
    assert websearch._max_bytes() == websearch.DEFAULT_MAX_BYTES
    monkeypatch.delenv("FETCH_MAX_BYTES")
    assert websearch._max_bytes() == websearch.DEFAULT_MAX_BYTES


# ───────────────────────────────────────────────────────────────────── AC-3.4
def test_binary_content_type_rejected_before_reading_body(monkeypatch) -> None:
    read = {"bytes": 0}

    def payload():
        read["bytes"] += 1024
        yield b"\x00" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/octet-stream"},
                              content=payload())

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    with pytest.raises(websearch.SearchError) as err:
        websearch.fetch_page("http://public.example/file.bin",
                             transport=transport(handler))

    assert "ไม่ใช่หน้าเว็บ" in str(err.value)
    assert read["bytes"] == 0, "อ่าน body ก่อนตรวจ content-type"


@pytest.mark.parametrize("ctype,ok", [
    ("text/html; charset=utf-8", True),
    ("text/plain", True),
    ("application/xhtml+xml", True),
    ("application/json", True),
    ("application/octet-stream", False),
    ("image/png", False),
    ("", False),
])
def test_content_type_allowlist(ctype, ok) -> None:
    assert websearch._content_type_ok(ctype) is ok


# ───────────────────────────────────────────────────────────────────── AC-3.5
def test_slow_drip_stops_within_total_budget(monkeypatch) -> None:
    def drip():
        for _ in range(10_000):
            time.sleep(0.02)
            yield b"<p>x</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML, content=drip())

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    t0 = time.monotonic()
    with pytest.raises(websearch.SearchError) as err:
        websearch.fetch_page("http://public.example/slow", timeout=1.0,
                             max_bytes=10 ** 9, transport=transport(handler))
    elapsed = time.monotonic() - t0

    assert "ไม่ทัน" in str(err.value)
    assert elapsed < 3.0, f"ใช้เวลา {elapsed:.1f} วินาที ทั้งที่งบเวลา 1 วินาที"


def test_redirect_request_uses_only_the_remaining_time_budget(monkeypatch) -> None:
    """redirect ใหม่ต้องไม่เริ่ม timeout เต็มก้อนอีกครั้ง"""
    read_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        read_timeouts.append(request.extensions["timeout"]["read"])
        if request.url.path == "/first":
            time.sleep(0.55)
            return httpx.Response(302, headers={"location": "/second"})
        return page()

    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    websearch.fetch_page(
        "http://public.example/first", timeout=1.0,
        transport=transport(handler))

    assert len(read_timeouts) == 2
    assert 0 < read_timeouts[1] < 0.6, read_timeouts


# ───────────────────────────────────────────────────────────────────── AC-3.6
@pytest.mark.parametrize("url,why", [
    ("http://public.example:22/", "พอร์ตนอก allowlist"),
    ("http://public.example:8080/", "พอร์ตนอก allowlist"),
    ("file:///etc/passwd", "ไม่ใช่ http"),
    ("ftp://public.example/x", "ไม่ใช่ http"),
    ("http://user:pw@public.example/", "ฝังรหัสผ่าน"),
    ("http:///nohost", "ไม่มีโฮสต์"),
])
def test_assert_fetchable_rejects(monkeypatch, url, why) -> None:
    monkeypatch.setattr(websearch, "_is_public", lambda host: True)

    def handler(request: httpx.Request) -> httpx.Response:   # pragma: no cover
        raise AssertionError("ต้องไม่มี request ออกไปเลย")

    with pytest.raises(websearch.SearchError):
        websearch.assert_fetchable(url)
    with pytest.raises(websearch.SearchError):
        websearch.fetch_page(url, transport=transport(handler))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:80/", "http://[::1]/", "http://192.168.1.1/",
    "http://10.0.0.5/", "http://169.254.169.254/latest/meta-data/",
    "http://[::ffff:127.0.0.1]/",
])
def test_private_addresses_rejected(url) -> None:
    with pytest.raises(websearch.SearchError):
        websearch.assert_fetchable(url)


def test_public_url_passes(monkeypatch) -> None:
    monkeypatch.setattr(websearch, "_is_public", lambda host: True)
    assert websearch.assert_fetchable("https://example.com/a?b=1") == \
        "https://example.com/a?b=1"
