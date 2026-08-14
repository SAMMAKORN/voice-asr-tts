"""P1-1 — WebSocket ต้องตรวจ Origin + token ก่อนรับการเชื่อมต่อ (กัน CSWSH)

ทั้งไฟล์ไม่ต่อเน็ตจริง: ใช้ TestClient ของ Starlette และสวม WebSession ด้วยของปลอม
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from html import escape

import pytest
import websockets
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from web import server

pytestmark = pytest.mark.unit

TOKEN = "test-token-1234567890abcdef"   # ต้องเป็น ASCII เหมือน token จริง


class FakeSession:
    """แทน WebSession ตัวจริง — ไม่แตะไมค์/ลำโพง/API"""

    created: list["FakeSession"] = []

    def __init__(self, cfg, out) -> None:
        self.cfg, self.out = cfg, out
        self.stopped = False
        FakeSession.created.append(self)

    def run(self) -> None:
        self.out.json("ready", chat_model=self.cfg.chat_model)

    def request_stop(self) -> None:
        self.stopped = True
        self.out.close()

    def close_resources(self) -> None:
        self.closed = True

    def feed_audio(self, data: bytes) -> None: ...

    def handle_client(self, msg: dict) -> None: ...


@pytest.fixture
def live_server(monkeypatch, cfg):
    """เซิร์ฟเวอร์ uvicorn จริงบน localhost — ไม่แตะ API จริง (WebSession ปลอม)

    ใช้เฉพาะเทสต์ที่ต้องให้ `finally` ของ /ws ทำงานจนจบ ซึ่ง TestClient ทำไม่ได้
    """
    import uvicorn

    monkeypatch.setenv("WEB_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("WEB_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(server, "_token", None)
    monkeypatch.setattr(server, "session_config", lambda: cfg)
    monkeypatch.setattr(server, "WebSession", FakeSession)
    monkeypatch.setattr(server, "_active", 0)
    server.LIMITER.reset()
    FakeSession.created.clear()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port,
                                        log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):                      # รอให้พอร์ตเปิดจริงก่อนคืนค่า
        try:
            with socket.create_connection(("127.0.0.1", port), 0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("เซิร์ฟเวอร์ทดสอบไม่ขึ้นภายในเวลาที่รอ")

    yield port

    srv.should_exit = True
    thread.join(10)


@pytest.fixture
def client(monkeypatch, cfg):
    monkeypatch.setenv("WEB_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("WEB_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    monkeypatch.setattr(server, "_token", None)          # ให้อ่าน env ใหม่
    monkeypatch.setattr(server, "session_config", lambda: cfg)
    monkeypatch.setattr(server, "WebSession", FakeSession)
    FakeSession.created.clear()
    server.STATS.update(sessions_created=0, rejected=0, rate_limited=0)
    # โควตาเป็น state ระดับโมดูล — ไม่ล้างแล้วเทสต์ก่อนหน้าจะทำให้เทสต์ถัดไปโดน 429
    server.LIMITER.reset()
    monkeypatch.setattr(server, "_active", 0)
    for key in ("WEB_RATE_LIMIT", "WEB_SELFTEST_LIMIT", "WEB_RATE_WINDOW_S",
                "WEB_MAX_SESSIONS", "WEB_TRUST_PROXY"):
        monkeypatch.delenv(key, raising=False)
    with TestClient(server.app) as c:
        yield c


# ───────────────────────────────────────────────────────────── AC-1.1 / AC-1.2
def test_ws_rejects_foreign_origin(client) -> None:
    """เว็บของผู้โจมตีต่อเข้ามา → ปิดด้วย 1008 และต้องไม่มี WebSession เกิดขึ้น"""
    with pytest.raises(WebSocketDisconnect) as err:
        with client.websocket_connect(
                f"/ws?token={TOKEN}",
                headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert err.value.code == 1008
    assert FakeSession.created == [], "สร้าง session ให้ origin ที่ไม่อนุญาต"
    assert server.STATS["sessions_created"] == 0
    assert server.STATS["rejected"] == 1


def test_ws_rejects_missing_origin_without_token(client) -> None:
    """client ที่ไม่ใช่เบราว์เซอร์ (ไม่มี Origin) ต้องมี token เท่านั้นถึงจะผ่าน"""
    with pytest.raises(WebSocketDisconnect) as err:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
    assert err.value.code == 1008
    assert server.STATS["sessions_created"] == 0

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?token=ผิด") as ws:
            ws.receive_json()
    assert server.STATS["sessions_created"] == 0


def test_ws_accepts_native_client_with_valid_token(client) -> None:
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "ready"
    assert server.STATS["sessions_created"] == 1


# ───────────────────────────────────────────────────── H-01 token มาทาง cookie
def test_index_sets_the_ws_token_cookie(client) -> None:
    """หน้าเว็บต้องแจก cookie ที่ /ws ใช้ยืนยันตัว — HttpOnly + SameSite=Strict"""
    res = client.get("/")
    raw = res.headers["set-cookie"]
    assert server.WS_TOKEN_COOKIE in raw
    assert "httponly" in raw.lower()
    assert "samesite=strict" in raw.lower()
    assert res.cookies[server.WS_TOKEN_COOKIE] == TOKEN


def test_ws_accepts_token_from_cookie_without_query_param(client) -> None:
    """เบราว์เซอร์ปกติ: โหลดหน้าเว็บ → ได้ cookie → ต่อ /ws โดยไม่มี ?token= เลย"""
    client.get("/")            # รับ cookie เข้า cookie jar เหมือนเบราว์เซอร์
    with client.websocket_connect(
            "/ws", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
        assert ws.receive_json()["type"] == "ready"
    assert server.STATS["sessions_created"] == 1
    assert server.STATS["rejected"] == 0


def test_ws_rejects_a_wrong_cookie(client) -> None:
    client.cookies.set(server.WS_TOKEN_COOKIE, "wrong-token")
    with pytest.raises(WebSocketDisconnect) as err:
        with client.websocket_connect(
                "/ws", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
            ws.receive_json()
    assert err.value.code == 1008
    assert server.STATS["sessions_created"] == 0
    assert server.STATS["rejected"] == 1


def test_cookie_carries_a_non_ascii_token(client, monkeypatch) -> None:
    """WEB_AUTH_TOKEN ที่เป็นภาษาไทยต้องใช้ได้จริง

    เคยพยายามส่ง token ทาง Sec-WebSocket-Protocol แล้วพังตรงนี้: ค่า subprotocol
    ต้องเป็น token ตาม RFC 7230 เบราว์เซอร์จึงโยน SyntaxError ทิ้งทั้งการเชื่อมต่อ
    """
    thai = "เทสต์-token"
    monkeypatch.setenv("WEB_AUTH_TOKEN", thai)
    monkeypatch.setattr(server, "_token", None)

    client.get("/")
    with client.websocket_connect(
            "/ws", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
        assert ws.receive_json()["type"] == "ready"
    assert server.STATS["sessions_created"] == 1


def test_secure_flag_follows_the_scheme(client) -> None:
    """ตั้ง Secure บน http จะทำให้เบราว์เซอร์ทิ้ง cookie แล้วต่อ /ws ไม่ได้เลย"""
    assert "secure" not in client.get("/").headers["set-cookie"].lower()
    res = client.get("/", headers={"X-Forwarded-Proto": "https"})
    assert "secure" in res.headers["set-cookie"].lower(), (
        "อยู่หลัง reverse proxy ที่ terminate TLS ต้องยังตั้ง Secure ให้")


# ───────────────────────────────────────────────────────────────────── AC-1.3
def test_ws_accepts_browser_from_localhost(client) -> None:
    """เปิดหน้าเว็บตามปกติแล้วกดเริ่มสนทนา → ต้องทำงานได้เหมือนเดิม"""
    for origin in ("http://127.0.0.1:8000", "http://localhost:8000"):
        with client.websocket_connect(f"/ws?token={TOKEN}",
                                      headers={"Origin": origin}) as ws:
            assert ws.receive_json()["type"] == "ready"
    assert server.STATS["sessions_created"] == 2
    assert server.STATS["rejected"] == 0


def test_allowed_origins_env_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("WEB_ALLOWED_ORIGINS", "https://voice.example.com, http://a.test")
    assert server.origin_allowed("https://voice.example.com", 8000) is True
    assert server.origin_allowed("http://a.test/", 8000) is True
    assert server.origin_allowed("http://127.0.0.1:8000", 8000) is False


def test_default_origin_rule_is_loopback_on_same_port(monkeypatch) -> None:
    monkeypatch.delenv("WEB_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    assert server.origin_allowed("http://127.0.0.1:8000", 8000) is True
    assert server.origin_allowed("http://localhost:8000", 8000) is True
    assert server.origin_allowed("http://127.0.0.1:9999", 8000) is False
    assert server.origin_allowed("http://192.168.1.20:8000", 8000) is False
    assert server.origin_allowed("null", 8000) is False          # iframe sandbox


# ───────────────────────────────────────────────────────────────────── AC-1.4
@pytest.mark.parametrize("method,path", [("get", "/api/config"),
                                         ("post", "/api/selftest")])
def test_api_requires_token(client, method, path) -> None:
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(
        path, headers={"X-Session-Token": "wrong-token"}).status_code == 401


def test_api_config_with_token_hides_internal_url(client) -> None:
    res = client.get("/api/config", headers={"X-Session-Token": TOKEN})
    assert res.status_code == 200
    body = res.json()
    assert "base_url" not in body and body["api_configured"] is True
    assert body["chat_model"] == "fake-chat"


# ─────────────────────────────────────────────────── M-02 จำกัดอัตราการเรียก
def test_selftest_is_rate_limited_per_client(client, monkeypatch) -> None:
    """/api/selftest ยิงครบวง TTS→ASR→LLM ทุกครั้ง — ต้องมีเพดานต่อผู้เรียก"""
    monkeypatch.setenv("WEB_SELFTEST_LIMIT", "3")
    monkeypatch.setattr(server, "run_selftest", lambda cfg: {"ok": True})
    head = {"X-Session-Token": TOKEN}

    codes = [client.post("/api/selftest", headers=head).status_code for _ in range(5)]

    assert codes[:3] == [200, 200, 200], codes
    assert codes[3:] == [429, 429], codes
    assert server.STATS["rate_limited"] == 2


def test_rate_limited_response_says_when_to_retry(client, monkeypatch) -> None:
    monkeypatch.setenv("WEB_SELFTEST_LIMIT", "1")
    monkeypatch.setenv("WEB_RATE_WINDOW_S", "60")
    monkeypatch.setattr(server, "run_selftest", lambda cfg: {"ok": True})
    head = {"X-Session-Token": TOKEN}

    assert client.post("/api/selftest", headers=head).status_code == 200
    res = client.post("/api/selftest", headers=head)

    assert res.status_code == 429
    assert 0 < int(res.headers["Retry-After"]) <= 61


def test_config_and_selftest_have_separate_budgets(client, monkeypatch) -> None:
    """ยิง /api/config จนเต็มโควตาต้องไม่ทำให้ /api/selftest ใช้ไม่ได้"""
    monkeypatch.setenv("WEB_RATE_LIMIT", "2")
    monkeypatch.setenv("WEB_SELFTEST_LIMIT", "2")
    monkeypatch.setattr(server, "run_selftest", lambda cfg: {"ok": True})
    head = {"X-Session-Token": TOKEN}

    for _ in range(3):
        client.get("/api/config", headers=head)
    assert client.get("/api/config", headers=head).status_code == 429
    assert client.post("/api/selftest", headers=head).status_code == 200


def test_limit_of_zero_disables_the_cap(client, monkeypatch) -> None:
    """ทางหนีสำหรับคนที่ไปจำกัดที่ reverse proxy เองแล้ว"""
    monkeypatch.setenv("WEB_RATE_LIMIT", "0")
    head = {"X-Session-Token": TOKEN}
    assert all(client.get("/api/config", headers=head).status_code == 200
               for _ in range(30))


def test_rate_limit_is_per_client_not_global(monkeypatch) -> None:
    """ผู้ใช้คนหนึ่งยิงรัวต้องไม่ทำให้คนอื่นถูกบล็อกไปด้วย"""
    server.LIMITER.reset()
    assert server.LIMITER.check("api", "1.1.1.1", 2, 60) == 0.0
    assert server.LIMITER.check("api", "1.1.1.1", 2, 60) == 0.0
    assert server.LIMITER.check("api", "1.1.1.1", 2, 60) > 0      # เต็มโควตาแล้ว
    assert server.LIMITER.check("api", "2.2.2.2", 2, 60) == 0.0   # คนละคน ยังผ่าน


def test_forwarded_for_is_ignored_unless_proxy_is_trusted(monkeypatch) -> None:
    """ถ้าเชื่อ X-Forwarded-For เสมอ ใครก็ปลอม header รีเซ็ตโควตาตัวเองได้"""
    class Conn:
        headers = {"x-forwarded-for": "9.9.9.9, 10.0.0.1"}
        client = type("C", (), {"host": "127.0.0.1"})()

    monkeypatch.delenv("WEB_TRUST_PROXY", raising=False)
    assert server.client_key(Conn()) == "127.0.0.1"

    monkeypatch.setenv("WEB_TRUST_PROXY", "1")
    assert server.client_key(Conn()) == "9.9.9.9"


def test_ws_rejects_when_the_session_cap_is_full(client, monkeypatch) -> None:
    """session ที่เปิดค้างไว้พร้อมกันมีเพดาน — เกินแล้วต้องปิดด้วย 1013"""
    monkeypatch.setenv("WEB_MAX_SESSIONS", "1")
    client.get("/")

    with client.websocket_connect(
            "/ws", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
        assert ws.receive_json()["type"] == "ready"
        assert server.active_sessions() == 1

        with pytest.raises(WebSocketDisconnect) as err:
            with client.websocket_connect(
                    "/ws", headers={"Origin": "http://127.0.0.1:8000"}) as second:
                second.receive_json()
        assert err.value.code == 1013


def test_session_slot_is_released_after_a_normal_close(live_server) -> None:
    """โควตารั่วทีละหนึ่งทุกครั้งที่ปิดแท็บ = ไม่กี่ชั่วโมงก็ต่อไม่ได้อีกเลย

    ต้องใช้เซิร์ฟเวอร์จริง: TestClient หยุดขับ event loop ของ endpoint ทันทีที่ออก
    จาก context ของ websocket ทำให้ `finally` ไม่ได้ทำงานจนจบ แล้วเทสต์จะฟ้องว่า
    โควตารั่วทั้งที่ของจริงไม่รั่ว (เคยหลงทางกับเรื่องนี้มาแล้ว)
    """
    async def go() -> list[int]:
        seen = []
        for _ in range(3):
            async with websockets.connect(
                    f"ws://127.0.0.1:{live_server}/ws?token={TOKEN}") as ws:
                await ws.recv()
                seen.append(server.active_sessions())
            for _ in range(100):
                if server.active_sessions() == 0:
                    break
                await asyncio.sleep(0.05)
            seen.append(server.active_sessions())
        return seen

    assert asyncio.run(go()) == [1, 0, 1, 0, 1, 0]


def test_bad_numeric_env_fails_fast_naming_the_key(monkeypatch) -> None:
    monkeypatch.setenv("WEB_RATE_LIMIT", "ไม่ใช่เลข")
    with pytest.raises(SystemExit) as err:
        server.api_rate_limit()
    assert "WEB_RATE_LIMIT" in str(err.value)


def test_out_of_range_numeric_env_is_clamped(monkeypatch) -> None:
    monkeypatch.setenv("WEB_RATE_WINDOW_S", "999999")
    assert server.rate_window() == 3600


# ───────────────────────────────────────────────────────────── H-02 / H-03
def test_security_headers_present_on_every_response(client) -> None:
    """H-02 — header ความปลอดภัยต้องติดมากับทุก response (หน้าเว็บ + /static + /api)"""
    for res in (client.get("/"),
                client.get("/static/app.js"),
                client.get("/api/config", headers={"X-Session-Token": TOKEN})):
        h = res.headers
        assert h["X-Frame-Options"] == "DENY"
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "microphone=(self)" in h["Permissions-Policy"]
        assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
        assert "max-age=" in h["Strict-Transport-Security"]


def test_csp_locks_down_script_sources(client) -> None:
    """สคริปต์ต้องมาจาก origin เดียวกันเท่านั้น ห้าม inline/eval

    ชุดเทสต์เบราว์เซอร์ผ่อน 'unsafe-eval' ให้ Playwright ได้ (ดู test_frontend_ui.py)
    ค่าที่ส่งให้ผู้ใช้จริงต้องไม่ผ่อนตาม
    """
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "'unsafe-eval'" not in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp, (
        "เปิด unsafe-inline ให้ script-src เท่ากับ CSP กัน XSS ไม่ได้อีกเลย")
    assert "object-src 'none'" in csp


def test_theme_bootstrap_is_not_inline(client) -> None:
    """สคริปต์ตั้งธีมต้องอยู่ไฟล์แยก ไม่งั้น CSP ต้องเปิด unsafe-inline ให้ทั้งหน้า"""
    page = client.get("/").text
    assert "/static/theme-init.js" in page
    assert "<script>" not in page, "ยังมี inline script ค้างอยู่ในหน้า"
    assert client.get("/static/theme-init.js").status_code == 200


def test_openapi_schema_is_not_exposed(client) -> None:
    """H-03 — /openapi.json ต้องไม่เปิดสาธารณะ (เดิมคืน 200 ให้ผังทั้ง API)"""
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_index_embeds_session_token(client) -> None:
    html = client.get("/").text
    assert server.TOKEN_PLACEHOLDER not in html
    assert f'content="{TOKEN}"' in html


def test_index_escapes_a_user_configured_token(client, monkeypatch) -> None:
    special = 'abc"&<script>window.pwned=1</script>'
    monkeypatch.setenv("WEB_AUTH_TOKEN", special)
    monkeypatch.setattr(server, "_token", None)

    page = client.get("/").text

    assert f'content="{escape(special, quote=True)}"' in page
    assert "<script>window.pwned=1</script>" not in page
    assert server.token_ok(special)


# ───────────────────────────────────────────────────────────────────── AC-1.5
def test_ready_payload_has_no_local_path_or_internal_url(web_session) -> None:
    web_session.cfg.base_url = "http://litellm.corp.internal:4000"
    web_session.log.dir = "/Users/someone/Projects/voice-asr-tts/logs/session-1"
    web_session.header()

    ready = web_session.out.of("ready")
    assert len(ready) == 1
    assert "log_dir" not in ready[0] and "base_url" not in ready[0]
    assert ready[0]["api_configured"] is True

    blob = json.dumps(ready[0], ensure_ascii=False)
    assert "/Users/" not in blob and "litellm" not in blob


# ───────────────────────────────────────────────────────────────────── AC-1.6
def test_startup_warning_only_when_exposed(monkeypatch) -> None:
    monkeypatch.delenv("WEB_AUTH_TOKEN", raising=False)
    assert server.startup_warning("127.0.0.1") is None
    warn = server.startup_warning("0.0.0.0")
    assert warn and "WARNING" in warn and "WEB_AUTH_TOKEN" in warn

    monkeypatch.setenv("WEB_AUTH_TOKEN", TOKEN)
    assert server.startup_warning("0.0.0.0") is None
