"""เว็บเซิร์ฟเวอร์: เสิร์ฟหน้าเว็บ + สะพาน WebSocket ไปยัง VoiceChat

รัน:  python3 -m web.server            (เปิด http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import threading
import time
import traceback
from collections.abc import Callable
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from fastapi import (Depends, FastAPI, Header, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vc.config import Config, load_config
from vc.tts_edge import THAI_VOICES, voice_label

from .bridge import Outbox
from .session import WebSession, run_selftest

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

log = logging.getLogger("voicechat.web")

app = FastAPI(title="Voice Link", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ─────────────────────────────────────────────── security headers (H-02)
# แอปนี้ไม่ตั้ง header ความปลอดภัยเลยมาก่อน ถูก embed ใน iframe/clickjack ได้
# และไม่มี CSP กันสคริปต์ภายนอก ตั้ง header กลางที่นี่ครั้งเดียวใช้กับทุก response
# HTTP (WebSocket handshake ไม่ผ่าน http middleware จึงไม่โดนแตะ)
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    # HSTS มีผลเฉพาะเมื่อโหลดผ่าน https — บน http (localhost dev) เบราว์เซอร์เมิน
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    # CSP: script-src เป็น 'self' ล้วน — สคริปต์ที่ถูกฝังเข้ามาจึงรันไม่ได้เลย
    # (theme bootstrap ย้ายไป /static/theme-init.js แล้วเพื่อการนี้)
    # style-src ยังต้องมี 'unsafe-inline' เพราะ index.html ใช้ style="..." อยู่หลายจุด
    # ซึ่งอันตรายน้อยกว่ากันมากเมื่อ script ถูกล็อกไว้แล้ว จะรัดต่อก็ต้องรื้อ markup
    # connect-src 'self' ครอบ ws:// ที่ origin เดียวกันด้วย (ยืนยันกับ Chromium แล้ว)
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "form-action 'self'; object-src 'none'"
    ),
}


@app.middleware("http")
async def add_security_headers(request, call_next):
    resp = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        resp.headers.setdefault(key, value)
    return resp

# ─────────────────────────────────────────────── สิทธิ์เข้าใช้งาน (P1-1: CSWSH)
# WebSocket ไม่อยู่ใต้ Same-Origin Policy เว็บใดก็ได้ที่ผู้ใช้เปิดค้างไว้จึงต่อเข้า
# ws://127.0.0.1:8000/ws ได้ตรง ๆ อ่านทุกคำพูดและสั่ง LLM ในนามผู้ใช้ได้
# จึงต้องกันสองชั้น: ตรวจ Origin ก่อน accept + ต้องมี token ที่ฝังมากับหน้าเว็บ
TOKEN_PLACEHOLDER = "__SESSION_TOKEN__"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# H-01: token เดินทางมากับ cookie ไม่ใช่ ?token= เพราะค่าใน query string ติดไป
# กับ access log ของ proxy/CDN (เช่น Cloudflare) และ Referer ส่วน cookie ไม่รั่ว
#
# เคยลองใส่ไว้ใน Sec-WebSocket-Protocol แล้วไม่เวิร์ค เก็บไว้เป็นบันทึกกันหลงทางซ้ำ:
#   1) ค่า subprotocol ต้องเป็น "token" ตาม RFC 7230 — WEB_AUTH_TOKEN ที่เป็น
#      ภาษาไทยหรือมี " < > ทำให้ new WebSocket() โยน SyntaxError ทิ้งทันที
#   2) RFC 6455 §4.1 บังคับว่าถ้า client เสนอ subprotocol มา เซิร์ฟเวอร์ต้องตอบ
#      กลับมาด้วย ไม่ตอบ = เบราว์เซอร์ล้ม handshake ("no response was received")
#      ทำให้ทุก proxy/เซิร์ฟเวอร์ที่ไม่ echo header นี้ทำหน้าเว็บพังหมด
# cookie ไม่มีข้อจำกัดทั้งสองข้อ และ SameSite=Strict ยังกัน CSWSH เพิ่มอีกชั้น
# นอกเหนือจากการตรวจ Origin (เว็บอื่นเรียก /ws จะไม่ได้ cookie ติดไปด้วยเลย)
WS_TOKEN_COOKIE = "vl_session_token"


def request_is_https(request) -> bool:
    """ตั้งแฟล็ก Secure ให้ cookie เฉพาะเมื่อมาทาง https จริง

    ตั้ง Secure บน http (เช่น localhost ตอน dev) เบราว์เซอร์จะทิ้ง cookie ทันที
    แล้วต่อ WebSocket ไม่ได้ — ต้องดู X-Forwarded-Proto ด้วยเพราะหลัง Cloudflare
    /reverse proxy scheme ที่เห็นตรงนี้เป็น http แม้ผู้ใช้เข้ามาทาง https
    """
    if request.url.scheme == "https":
        return True
    fwd = request.headers.get("x-forwarded-proto", "")
    return fwd.split(",")[0].strip().lower() == "https"


_token_lock = threading.Lock()
_token: str | None = None

# token หมุนใหม่ทุกครั้งที่เสิร์ฟหน้าเว็บ (หนึ่งครั้งที่โหลดหน้า = หนึ่ง token)
# เดิมเป็น token ตัวเดียวต่อ process: รั่วครั้งเดียวก็ใช้ได้จนกว่าจะรีสตาร์ตเซิร์ฟเวอร์
# ต้องเก็บได้หลายตัวพร้อมกัน ไม่ใช่แทนที่ตัวเก่า เพราะผู้ใช้เปิดหลายแท็บได้และแต่ละแท็บ
# ถือ token ของตัวเอง — ถ้าโหลดแท็บใหม่แล้วตัวเก่าตายทันที แท็บที่เปิดค้างจะหลุดหมด
_live_lock = threading.Lock()            # แยกจาก _token_lock: Lock ไม่ reentrant
_live: dict[str, float] = {}             # token → เวลาหมดอายุ (นาฬิกา monotonic)

# เพดานจำนวน token ที่ถืออยู่พร้อมกัน — GET / ไม่ต้องมีสิทธิ์ ใครยิงรัวก็สร้าง token
# ได้เรื่อย ๆ ถ้าไม่มีเพดาน dict จะโตไม่จำกัด (ตั้งสูงพอที่การใช้งานจริงไม่มีทางแตะ)
TOKEN_MAX = 2048

# ตัวนับไว้ยืนยันใน test/log ว่า connection ที่ถูกปฏิเสธไม่ได้สร้าง WebSession จริง
STATS = {"sessions_created": 0, "rejected": 0, "rate_limited": 0}

# จำนวน session ที่กำลังทำงานอยู่ — แต่ละตัวกิน 1 เธรด + เปิดสิทธิ์เรียก API ทั้งชุด
_active_lock = threading.Lock()
_active = 0


def fixed_token() -> str | None:
    """token คงที่ที่ผู้ดูแลตั้งเอง (WEB_AUTH_TOKEN) — ตั้งไว้แล้วจะไม่หมุนเลย

    จำเป็นเมื่อรันหลาย worker: token ที่หมุนถูกเก็บในหน่วยความจำของ process เดียว
    worker ตัวอื่นจึงยืนยันไม่ผ่าน การตั้งค่านี้คือการเลือก "คงที่แต่ใช้ร่วมกันได้"
    แทน "หมุนแต่ใช้ได้ process เดียว" — ไม่ใช่ค่าที่ลืมตั้งแล้วเสียความปลอดภัย
    """
    global _token
    with _token_lock:
        if _token is None:
            _token = (os.environ.get("WEB_AUTH_TOKEN") or "").strip()
        return _token or None


def _prune_tokens(now: float) -> None:
    """ทิ้ง token ที่หมดอายุ — ต้องถูกเรียกใต้ `_live_lock` เสมอ"""
    for dead in [t for t, exp in _live.items() if exp <= now]:
        del _live[dead]


def issue_token() -> str:
    """แจก token ให้หนึ่ง session — เรียกตอนเสิร์ฟหน้าเว็บเท่านั้น

    ตั้ง WEB_AUTH_TOKEN ไว้ = คืนค่านั้นทุกครั้ง (ไม่หมุน ดู `fixed_token`)
    """
    fixed = fixed_token()
    if fixed:
        return fixed
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _live_lock:
        _prune_tokens(now)
        while _live and len(_live) >= TOKEN_MAX:   # `_live and` — TOKEN_MAX=0 ไม่วนค้าง
            # เต็มเพดาน: เตะตัวที่จะหมดอายุก่อนออกไป (= ตัวที่ไม่ได้ถูกใช้นานสุด
            # เพราะทุกครั้งที่ยืนยันผ่าน อายุถูกต่อใหม่) ยิง GET / รัว ๆ ครบเพดาน
            # ยังเตะแท็บที่เปิดค้างอยู่ได้ แต่คนที่เรียก / ได้ก็ได้ token ที่ใช้งาน
            # ได้อยู่แล้ว จึงไม่ได้สิทธิ์เพิ่ม — ผลเสียคือแท็บนั้นต้องโหลดหน้าใหม่
            del _live[min(_live, key=_live.get)]     # type: ignore[arg-type]
        _live[token] = now + token_ttl()
    return token


def token_ok(token: str | None) -> bool:
    if not token:
        return False
    # เทียบเป็น bytes เสมอ — compare_digest ของ str รับเฉพาะ ASCII
    # (client ส่ง token ภาษาไทยเข้ามาต้องได้ False ไม่ใช่ 500)
    raw = str(token).encode("utf-8")
    fixed = fixed_token()
    if fixed and secrets.compare_digest(raw, fixed.encode("utf-8")):
        return True
    now = time.monotonic()
    ttl = token_ttl()
    with _live_lock:
        _prune_tokens(now)
        for live in list(_live):
            if secrets.compare_digest(raw, live.encode("utf-8")):
                # ต่ออายุทุกครั้งที่ใช้งานได้จริง (sliding) — เซสชันที่ยังคุยอยู่จึง
                # ไม่หมดอายุกลางทาง ส่วนแท็บที่ถูกปิดทิ้งไว้จะหลุดไปเองตาม TTL
                _live[live] = now + ttl
                return True
    return False


def live_token_count() -> int:
    """จำนวน token ที่ยังใช้ได้ — ไว้ยืนยันใน test ว่าหมุนจริงและถูกกวาดจริง"""
    with _live_lock:
        _prune_tokens(time.monotonic())
        return len(_live)


def allowed_origins() -> list[str]:
    """allowlist จาก env `WEB_ALLOWED_ORIGINS` (คั่นด้วยจุลภาค)

    ไม่ตั้ง = ใช้กติกาปริยาย: ยอมเฉพาะ origin ที่เป็น loopback บนพอร์ตเดียวกับ
    ที่เซิร์ฟเวอร์ตัวนี้ให้บริการอยู่ (http://127.0.0.1:{port} / http://localhost:{port})
    """
    raw = os.environ.get("WEB_ALLOWED_ORIGINS") or ""
    return [o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()]


def origin_allowed(origin: str | None, port: int | None = None) -> bool:
    """ตรวจ Origin header — None (ไม่มี header) ถือว่า "ไม่ใช่เบราว์เซอร์" ผ่านชั้นนี้ไป
    แล้วไปตกที่ชั้น token แทน (ดู `ws_authorized`)
    """
    if origin is None:
        return True
    allow = allowed_origins()
    value = origin.strip().rstrip("/").lower()
    if allow:
        return value in allow
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if parts.hostname not in LOOPBACK_HOSTS:
        return False
    want = port or int(os.environ.get("WEB_PORT") or 0) or None
    if want is None:
        return True
    return (parts.port or (443 if parts.scheme == "https" else 80)) == want


def ws_authorized(origin: str | None, token: str | None, port: int | None = None) -> bool:
    """เงื่อนไขผ่าน: Origin อยู่ใน allowlist (ถ้ามี header) **และ** token ถูกต้อง"""
    return origin_allowed(origin, port) and token_ok(token)


def ws_client_token(ws: WebSocket) -> str | None:
    """token จาก handshake — อ่าน cookie ก่อน (ไม่รั่วเข้า log/Referer)
    ถ้าไม่มีค่อยตกไปอ่าน ?token= ให้ client แบบ native (curl/สคริปต์เทสต์)
    """
    raw = ws.cookies.get(WS_TOKEN_COOKIE)
    if raw:
        return unquote(raw)      # ตั้งค่าไว้เป็น percent-encoded เสมอ (ดู index())
    return ws.query_params.get("token")


def require_token(x_session_token: str | None = Header(default=None)) -> None:
    """สิทธิ์เข้าถึง /api/* — หน้าเว็บแนบ header นี้ให้อัตโนมัติ"""
    if not token_ok(x_session_token):
        raise HTTPException(status_code=401, detail="ต้องมี session token ที่ถูกต้อง")


# ─────────────────────────────────────────────── จำกัดอัตราการเรียก (M-02)
# /api/selftest ยิงครบวง TTS→ASR→LLM ทุกครั้ง (ราว 5-6 วินาทีและมีค่า API จริง)
# ใครถือ token อยู่ก็ยิงรัว ๆ ได้ไม่จำกัด = เผา GPU/เครดิตขององค์กรฟรี ๆ
# ทำเป็น sliding window ในตัว ไม่เพิ่ม dependency ให้โปรเจกต์
def env_int(name: str, default: int, low: int, high: int) -> int:
    """อ่านค่าจำนวนเต็มจาก env แบบเดียวกับ vc/config.py — ผิดชนิดล้มทันที เกินช่วงถูกหั่น"""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} ต้องเป็นจำนวนเต็ม (ได้ {raw!r}) "
                         f"ช่วงที่รับคือ {low}–{high}")
    if not (low <= value <= high):
        clamped = max(low, min(high, value))
        log.warning("%s=%s อยู่นอกช่วง %d–%d — ใช้ %d แทน",
                    name, value, low, high, clamped)
        return clamped
    return value


# ช่วงที่รับได้ของ key ตัวเลขฝั่งเว็บ — ประกาศเป็น dict เพื่อให้ทั้งโค้ดและเทสต์
# (tests/test_config_validation.py) อ่านจากที่เดียวกัน ไม่ต้องไล่แก้สองที่
# key เหล่านี้อยู่ที่นี่ไม่ใช่ vc/config.py เพราะเป็นเรื่องของเว็บล้วน ๆ และ vc/
# ต้องไม่รู้จักโหมดเว็บ (เหมือน WEB_ALLOWED_ORIGINS / WEB_AUTH_TOKEN)
WEB_RANGES: dict[str, tuple[int, int]] = {
    "WEB_RATE_WINDOW_S": (1, 3600),
    "WEB_RATE_LIMIT": (0, 100_000),
    "WEB_SELFTEST_LIMIT": (0, 100_000),
    "WEB_MAX_SESSIONS": (0, 10_000),
    # อย่างน้อย 1 นาที (สั้นกว่านั้นหน้าเว็บที่เพิ่งโหลดก็ต่อ /ws ไม่ทัน)
    # อย่างมาก 7 วัน — ไม่ใช่ 0 = ไม่หมดอายุ เพราะจุดประสงค์ของการหมุนคือให้มันหมดอายุ
    "WEB_TOKEN_TTL_S": (60, 604_800),
}


def rate_window() -> int:
    return env_int("WEB_RATE_WINDOW_S", 60, *WEB_RANGES["WEB_RATE_WINDOW_S"])


def token_ttl() -> int:
    """อายุ token ที่หมุน — นับจากครั้งล่าสุดที่ใช้งานได้ ไม่ใช่จากตอนแจก

    ค่าปริยาย 12 ชั่วโมง: ยาวพอครอบการใช้งานต่อเนื่องทั้งวันโดยไม่ต้องโหลดหน้าใหม่
    (การต่อ /ws และการเรียก /api/* นับเป็นการใช้งาน จึงต่ออายุให้เอง)
    """
    return env_int("WEB_TOKEN_TTL_S", 43_200, *WEB_RANGES["WEB_TOKEN_TTL_S"])


def api_rate_limit() -> int:
    return env_int("WEB_RATE_LIMIT", 60, *WEB_RANGES["WEB_RATE_LIMIT"])


def selftest_rate_limit() -> int:
    return env_int("WEB_SELFTEST_LIMIT", 5, *WEB_RANGES["WEB_SELFTEST_LIMIT"])


def max_sessions() -> int:
    return env_int("WEB_MAX_SESSIONS", 8, *WEB_RANGES["WEB_MAX_SESSIONS"])


def trust_proxy() -> bool:
    return (os.environ.get("WEB_TRUST_PROXY") or "").strip() in ("1", "true", "yes")


def client_key(conn) -> str:
    """ตัวระบุผู้เรียกสำหรับนับโควตา

    หลัง Cloudflare/reverse proxy ทุกคำขอมาจาก IP ของ proxy ตัวเดียว ถ้านับตามนั้น
    โควตาจะกลายเป็นของรวมทุกคนแล้วผู้ใช้จริงโดนบล็อกกันเอง — ตั้ง WEB_TRUST_PROXY=1
    เพื่อให้เชื่อ X-Forwarded-For (เชื่อได้เฉพาะเมื่ออยู่หลัง proxy จริง ไม่งั้น
    ใครก็ปลอม header นี้เพื่อรีเซ็ตโควตาตัวเองได้)
    """
    if trust_proxy():
        first = (conn.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if first:
            return first
    return conn.client.host if conn.client else "?"


class RateLimiter:
    """sliding window ต่อ (ชนิดคำขอ, ผู้เรียก) — ใช้ monotonic กันเวลาระบบถูกปรับ"""

    PRUNE_EVERY = 256        # กวาด key ที่ไม่มีใครใช้ทิ้ง กัน dict โตไม่จำกัด

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def check(self, kind: str, who: str, limit: int, window: int) -> float:
        """ผ่าน → 0.0 ; ไม่ผ่าน → จำนวนวินาทีที่ควรรอก่อนลองใหม่"""
        if limit <= 0:
            return 0.0          # 0 = ปิดการจำกัด
        now = time.monotonic()
        with self._lock:
            self._calls += 1
            if self._calls % self.PRUNE_EVERY == 0:
                self._prune(now, window)
            hits = self._hits.setdefault((kind, who), [])
            hits[:] = [t for t in hits if t > now - window]
            if len(hits) >= limit:
                return max(0.01, hits[0] + window - now)
            hits.append(now)
            return 0.0

    def _prune(self, now: float, window: int) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - window]:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._calls = 0


LIMITER = RateLimiter()


def rate_limited(kind: str, limit: Callable[[], int]):
    """สร้าง dependency สำหรับ /api/* — เกินโควตาได้ 429 พร้อม Retry-After"""
    def dep(request: Request) -> None:
        wait = LIMITER.check(kind, client_key(request), limit(), rate_window())
        if wait:
            STATS["rate_limited"] += 1
            raise HTTPException(
                status_code=429,
                detail=f"เรียกถี่เกินไป ลองใหม่ในอีก {int(wait) + 1} วินาที",
                headers={"Retry-After": str(int(wait) + 1)})
    return dep


def startup_warning(host: str) -> str | None:
    """ข้อความเตือนตอนเปิดเซิร์ฟเวอร์ให้ทั้งวง LAN เข้าถึงได้โดยไม่ตั้ง token เอง"""
    if host in ("127.0.0.1", "localhost", "::1"):
        return None
    if (os.environ.get("WEB_AUTH_TOKEN") or "").strip():
        return None
    return (
        "  ⚠️  WARNING: เปิดให้เครื่องอื่นในวง LAN เข้าถึงได้ "
        f"(--host {host}) โดยยังไม่ได้ตั้ง WEB_AUTH_TOKEN\n"
        "      ใครก็ตามที่เปิดหน้าเว็บนี้ได้จะได้ token ไปด้วย และใช้ API key "
        "ขององค์กรผ่านเครื่องนี้ได้\n"
        "      วิธีปิดความเสี่ยง: ตั้ง WEB_AUTH_TOKEN=<ค่าลับ> และ "
        "WEB_ALLOWED_ORIGINS=<origin ที่อนุญาต> ใน .env\n"
        "      หรือใช้ค่าเริ่มต้น --host 127.0.0.1 แล้วต่อผ่าน SSH tunnel แทน"
    )


def session_config() -> Config:
    """คอนฟิกใหม่ต่อหนึ่ง session (แต่ละแท็บปรับ mute/echo แยกกันได้)

    เปิด echo guard ไว้ทั้งที่เบราว์เซอร์มี AEC ของตัวเองแล้ว เพราะจากบันทึกจริง
    ลำโพงโน้ตบุ๊กยังรั่วเข้าไมค์พอให้ ASR เดาเป็นข้อความมั่ว ๆ แล้วไปตัดคำตอบทิ้ง
    (ปิดได้จากปุ่มในหน้าเว็บถ้าใส่หูฟัง จะพูดแทรกได้ไวขึ้น)
    """
    return load_config()


def voice_options(cfg: Config) -> list[dict]:
    """ตัวเลือกเสียงพูดสำหรับ dropdown บนหน้าเว็บ

    ค่า value ใช้รูป "api" หรือ "edge:<ชื่อเสียง>" เพื่อส่งกลับมาทาง WebSocket
    ได้ทั้งก้อนเดียว ไม่ต้องแยกเป็นสองฟิลด์
    """
    items: list[dict] = []
    if cfg.tts_model:
        # ใช้ cfg.tts_label ไม่ใช่ cfg.tts_model ตรง ๆ — ประกอบชื่อเองตรงนี้ทำให้
        # dropdown ไม่รู้เรื่องการโคลนเสียง แล้วโชว์คนละอย่างกับ chip บนหัวเว็บ
        items.append({"value": "api", "label": f"เซิร์ฟเวอร์ · {cfg.tts_label}"})
    names = [name for name, _label, _gender in THAI_VOICES]
    if cfg.edge_voice and cfg.edge_voice not in names:
        names.append(cfg.edge_voice)     # เสียงที่ตั้งเองใน .env ต้องเลือกกลับมาได้
    for name in names:
        items.append({"value": f"edge:{name}", "label": f"Edge · {voice_label(name)}"})
    return items


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    """ฝัง session token ลงหน้าเว็บตอนเสิร์ฟ — client แนบกลับมาทุกคำขอ

    meta tag ใช้กับ /api/* (header X-Session-Token) ส่วน cookie ใช้กับ /ws
    (H-01 — WebSocket ตั้ง header เองไม่ได้ แต่ cookie ถูกแนบให้อัตโนมัติ)

    token หมุนใหม่ทุกครั้งที่โหลดหน้า — ต้องแจกครั้งเดียวแล้วใช้ค่าเดียวกันทั้ง meta
    และ cookie (เรียก issue_token() สองรอบ = ได้สองค่า แล้ว /ws กับ /api/* จะถือ
    token คนละตัว ซึ่งใช้ได้ทั้งคู่แต่เปลืองโควตาใน store ไปเปล่า ๆ ทุกการโหลดหน้า)
    """
    token = issue_token()
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace(TOKEN_PLACEHOLDER, escape(token, quote=True))
    resp = HTMLResponse(html, headers={"Cache-Control": "no-store"})
    resp.set_cookie(
        # ค่า cookie ต้อง encode เป็น latin-1 ได้ WEB_AUTH_TOKEN ที่เป็นภาษาไทย
        # ทำให้ set_cookie โยน UnicodeEncodeError แล้วหน้าเว็บกลายเป็น 500 ทั้งหน้า
        WS_TOKEN_COOKIE, quote(token, safe=""),
        httponly=True,          # JS อ่านไม่ได้ ลด surface ตอนมี XSS
        samesite="strict",      # เว็บอื่นเรียก /ws จะไม่ได้ cookie ไปด้วย (กัน CSWSH)
        secure=request_is_https(request),
        path="/",
    )
    return resp


@app.get("/api/config", dependencies=[Depends(require_token),
                                      Depends(rate_limited("config", api_rate_limit))])
async def api_config() -> JSONResponse:
    cfg = session_config()
    return JSONResponse({
        "chat_model": cfg.chat_model,
        "asr_model": cfg.asr_model,
        "tts_model": cfg.tts_model,
        "tts_label": cfg.tts_label,
        "tts_voice": cfg.tts_voice_value,
        "tts_voices": voice_options(cfg),
        # ไม่ส่ง base_url ออกไป — เป็น endpoint ภายในองค์กร (P1-1 ข้อ 4)
        "api_configured": bool(cfg.base_url and cfg.api_key),
        "mic_sr": cfg.mic_sr,
        "speaker_sr": cfg.speaker_sr,
        "frame_ms": cfg.frame_ms,
    })


@app.post("/api/selftest",
          dependencies=[Depends(require_token),
                        Depends(rate_limited("selftest", selftest_rate_limit))])
async def api_selftest(voice: str = "") -> JSONResponse:
    cfg = session_config()
    if voice:
        cfg.set_tts_voice(voice)   # ค่าผิดรูปถูกเมิน แล้วตกกลับไปใช้ค่าใน .env
    result = await asyncio.to_thread(run_selftest, cfg)
    return JSONResponse(result)


async def _pump(ws: WebSocket, out: Outbox) -> None:
    """ดึงข้อความที่ session สร้างไว้ (คนละเธรด) ออกไปให้เบราว์เซอร์"""
    while True:
        item = await out.queue.get()
        if item is None:
            return
        kind, payload = item
        try:
            if kind == "json":
                await ws.send_json(payload)
            else:
                await ws.send_bytes(payload)  # type: ignore[arg-type]
        except Exception:       # client หลุดไปแล้ว
            return


JOIN_TIMEOUT = 5.0       # วินาทีที่ยอมรอเธรด session ปิดตัว


def _thread_stack(thread: threading.Thread) -> str:
    frames = sys._current_frames().get(thread.ident or -1)
    if frames is None:
        return "(ไม่พบ stack ของเธรดนี้)"
    return "".join(traceback.format_stack(frames))


def release_session_slot() -> None:
    """คืนโควตา session ที่จองไว้ — เรียกได้ทุกทางออกของ /ws (ห้ามติดลบ)"""
    global _active
    with _active_lock:
        _active = max(0, _active - 1)


def active_sessions() -> int:
    with _active_lock:
        return _active


async def close_session(session, worker: threading.Thread) -> None:
    """ปิด session ให้จบจริง — ถ้า join ไม่สำเร็จต้องดังและต้องไม่ทิ้งทรัพยากรค้าง"""
    session.request_stop()
    await asyncio.to_thread(worker.join, JOIN_TIMEOUT)
    if worker.is_alive():
        log.error("join timeout: เธรด %r ไม่จบภายใน %.1f วินาที "
                  "— ปิดทรัพยากรให้เองแทน\n%s",
                  worker.name, JOIN_TIMEOUT, _thread_stack(worker))
    # เรียกเสมอ แม้ join สำเร็จ (ซ้ำได้ ไม่มีผลข้างเคียง) เพื่อรับประกันว่า
    # log.close()/api.close() ถูกเรียกแม้เธรด session จะค้างจน shutdown() ไม่ทำงาน
    session.close_resources()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, voice: str = "") -> None:
    # ตรวจสิทธิ์ "ก่อน" accept เสมอ — ถ้า accept ไปแล้วค่อยปิด ผู้โจมตีจะได้
    # connection ที่เปิดจริงชั่วขณะและอาจได้ข้อความแรกไปด้วย
    origin = ws.headers.get("origin")
    token = ws_client_token(ws)
    if not ws_authorized(origin, token, ws.url.port):
        STATS["rejected"] += 1
        log.warning("ปฏิเสธการเชื่อมต่อ WebSocket: origin=%r token=%s",
                    origin, "ถูกต้อง" if token_ok(token) else "ผิด/ไม่มี")
        await ws.close(code=1008)
        return

    # M-02: กันเปิด session รัว ๆ ทั้งแบบถี่และแบบค้างไว้เยอะ ๆ พร้อมกัน
    # ต้องเช็คก่อน accept เหมือนชั้นตรวจสิทธิ์ ไม่ให้ได้ socket ที่เปิดจริงมาก่อน
    wait = LIMITER.check("ws", client_key(ws), api_rate_limit(), rate_window())
    if wait:
        STATS["rate_limited"] += 1
        log.warning("ปฏิเสธ WebSocket: เปิด session ถี่เกินไปจาก %s", client_key(ws))
        await ws.close(code=1013)      # 1013 = Try Again Later
        return

    cap = max_sessions()
    with _active_lock:
        global _active
        if cap and _active >= cap:
            STATS["rate_limited"] += 1
            log.warning("ปฏิเสธ WebSocket: session ที่เปิดอยู่ครบ %d แล้ว", cap)
            await ws.close(code=1013)
            return
        _active += 1

    try:
        await ws.accept()
    except Exception:
        release_session_slot()
        raise
    loop = asyncio.get_running_loop()
    out = Outbox(loop)

    try:
        cfg = session_config()
    except SystemExit as exc:
        await ws.send_json({"type": "log", "level": "error", "text": str(exc)})
        await ws.close()
        release_session_slot()
        return

    if voice:
        # เบราว์เซอร์ส่งเสียงที่จำไว้มาตั้งแต่ตอนต่อ WebSocket — ต้องตั้งค่า cfg
        # ให้เสร็จก่อน WebSession เริ่มทักทาย ไม่งั้นทักทายไปด้วยเสียง/เพศจาก .env
        # ก่อนที่ฝั่งเว็บจะทันส่งคำสั่งสลับเสียงมาทีหลัง (แข่งกันตอนเริ่ม session)
        cfg.set_tts_voice(voice)   # ค่าผิดรูปถูกเมิน แล้วตกกลับไปใช้ค่าใน .env

    session = WebSession(cfg, out)
    STATS["sessions_created"] += 1
    worker = threading.Thread(target=session.run, name="session", daemon=True)
    worker.start()
    pump = asyncio.create_task(_pump(ws, out))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            # ข้อความเดียวที่ผิดรูปต้องไม่ล้มทั้ง session — session ตรวจ field เองอีกชั้น
            # (P2-11) ส่วนตรงนี้กันกรณีที่หลุดออกมาถึงชั้น transport จริง ๆ
            data = msg.get("bytes")
            if data:
                try:
                    session.feed_audio(data)
                except Exception as exc:  # noqa: BLE001
                    log.warning("ข้ามเฟรมเสียงที่ผิดรูป: %r", exc)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                log.debug("ข้ามข้อความที่ไม่ใช่ JSON (%d ไบต์)", len(text))
                continue
            if not isinstance(payload, dict):
                continue
            try:
                session.handle_client(payload)
            except Exception as exc:  # noqa: BLE001
                log.warning("ข้ามข้อความควบคุมที่ผิดรูป: %r", exc)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error: %r", exc)
    finally:
        # คืนโควตาก่อน await ที่เหลือ — ถ้าไปวางท้ายสุดแล้วมี await ตัวใดค้าง
        # (เคยเจอ pump ค้างใต้ TestClient) โควตาจะรั่วถาวรทีละหนึ่งต่อการปิดแท็บ
        release_session_slot()
        await close_session(session, worker)
        out.close()
        await pump
        try:
            await ws.close()
        except Exception:
            pass


def main() -> int:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="เว็บแอปคุยกับ AI ด้วยเสียง")
    p.add_argument("--host", default="127.0.0.1", help="ค่าเริ่มต้น 127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="รีโหลดอัตโนมัติตอนแก้โค้ด")
    args = p.parse_args()

    load_config()   # ให้ล้มตั้งแต่ตอนเปิดถ้า .env ไม่ครบ
    # ให้ worker process ของ --reload เห็นพอร์ตเดียวกัน (ใช้คำนวณ origin ปริยาย)
    os.environ["WEB_HOST"] = args.host
    os.environ["WEB_PORT"] = str(args.port)
    print(f"\n  เปิดเบราว์เซอร์ที่  http://{args.host}:{args.port}\n")
    warn = startup_warning(args.host)
    if warn:
        print(warn + "\n")
    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning",
                # ไม่ประกาศ "Server: uvicorn" ให้ผู้ที่กำลังสำรวจช่องโหว่อ่านฟรี ๆ
                # ต้องปิดที่ชั้น uvicorn: มันเอา default header ไปต่อกับ header ที่แอป
                # ส่ง ไม่ใช่แทนที่ ฉะนั้นถ้าไปตั้ง Server เองใน middleware จะได้ header
                # ซ้ำสองอันแทนที่จะทับกัน (SECURITY_HEADERS จึงช่วยเรื่องนี้ไม่ได้)
                server_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
