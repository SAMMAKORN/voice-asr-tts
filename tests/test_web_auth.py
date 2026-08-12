"""P1-1 — WebSocket ต้องตรวจ Origin + token ก่อนรับการเชื่อมต่อ (กัน CSWSH)

ทั้งไฟล์ไม่ต่อเน็ตจริง: ใช้ TestClient ของ Starlette และสวม WebSession ด้วยของปลอม
"""
from __future__ import annotations

import json

import pytest
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

    def feed_audio(self, data: bytes) -> None: ...

    def handle_client(self, msg: dict) -> None: ...


@pytest.fixture
def client(monkeypatch, cfg):
    monkeypatch.setenv("WEB_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("WEB_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    monkeypatch.setattr(server, "_token", None)          # ให้อ่าน env ใหม่
    monkeypatch.setattr(server, "session_config", lambda: cfg)
    monkeypatch.setattr(server, "WebSession", FakeSession)
    FakeSession.created.clear()
    server.STATS.update(sessions_created=0, rejected=0)
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


def test_index_embeds_session_token(client) -> None:
    html = client.get("/").text
    assert server.TOKEN_PLACEHOLDER not in html
    assert f'content="{TOKEN}"' in html


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
