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
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vc.config import Config, load_config

from .bridge import Outbox
from .session import WebSession, run_selftest

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

log = logging.getLogger("voicechat.web")

app = FastAPI(title="Voice Link", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ─────────────────────────────────────────────── สิทธิ์เข้าใช้งาน (P1-1: CSWSH)
# WebSocket ไม่อยู่ใต้ Same-Origin Policy เว็บใดก็ได้ที่ผู้ใช้เปิดค้างไว้จึงต่อเข้า
# ws://127.0.0.1:8000/ws ได้ตรง ๆ อ่านทุกคำพูดและสั่ง LLM ในนามผู้ใช้ได้
# จึงต้องกันสองชั้น: ตรวจ Origin ก่อน accept + ต้องมี token ที่ฝังมากับหน้าเว็บ
TOKEN_PLACEHOLDER = "__SESSION_TOKEN__"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_token_lock = threading.Lock()
_token: str | None = None

# ตัวนับไว้ยืนยันใน test/log ว่า connection ที่ถูกปฏิเสธไม่ได้สร้าง WebSession จริง
STATS = {"sessions_created": 0, "rejected": 0}


def session_token() -> str:
    """token ประจำ process — ตั้ง WEB_AUTH_TOKEN เองได้เมื่อรันหลาย worker"""
    global _token
    with _token_lock:
        if _token is None:
            _token = (os.environ.get("WEB_AUTH_TOKEN") or "").strip() \
                or secrets.token_urlsafe(32)
        return _token


def token_ok(token: str | None) -> bool:
    if not token:
        return False
    # เทียบเป็น bytes เสมอ — compare_digest ของ str รับเฉพาะ ASCII
    # (client ส่ง token ภาษาไทยเข้ามาต้องได้ False ไม่ใช่ 500)
    return secrets.compare_digest(str(token).encode("utf-8"),
                                  session_token().encode("utf-8"))


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


def require_token(x_session_token: str | None = Header(default=None)) -> None:
    """สิทธิ์เข้าถึง /api/* — หน้าเว็บแนบ header นี้ให้อัตโนมัติ"""
    if not token_ok(x_session_token):
        raise HTTPException(status_code=401, detail="ต้องมี session token ที่ถูกต้อง")


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


@app.get("/")
async def index() -> HTMLResponse:
    """ฝัง session token ลงหน้าเว็บตอนเสิร์ฟ — client แนบกลับมาทุกคำขอ"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace(TOKEN_PLACEHOLDER, session_token())
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/config", dependencies=[Depends(require_token)])
async def api_config() -> JSONResponse:
    cfg = session_config()
    return JSONResponse({
        "chat_model": cfg.chat_model,
        "asr_model": cfg.asr_model,
        "tts_model": cfg.tts_model,
        # ไม่ส่ง base_url ออกไป — เป็น endpoint ภายในองค์กร (P1-1 ข้อ 4)
        "api_configured": bool(cfg.base_url and cfg.api_key),
        "mic_sr": cfg.mic_sr,
        "speaker_sr": cfg.speaker_sr,
        "frame_ms": cfg.frame_ms,
    })


@app.post("/api/selftest", dependencies=[Depends(require_token)])
async def api_selftest() -> JSONResponse:
    result = await asyncio.to_thread(run_selftest, session_config())
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
async def ws_endpoint(ws: WebSocket) -> None:
    # ตรวจสิทธิ์ "ก่อน" accept เสมอ — ถ้า accept ไปแล้วค่อยปิด ผู้โจมตีจะได้
    # connection ที่เปิดจริงชั่วขณะและอาจได้ข้อความแรกไปด้วย
    origin = ws.headers.get("origin")
    token = ws.query_params.get("token")
    if not ws_authorized(origin, token, ws.url.port):
        STATS["rejected"] += 1
        log.warning("ปฏิเสธการเชื่อมต่อ WebSocket: origin=%r token=%s",
                    origin, "ถูกต้อง" if token_ok(token) else "ผิด/ไม่มี")
        await ws.close(code=1008)
        return

    await ws.accept()
    loop = asyncio.get_running_loop()
    out = Outbox(loop)

    try:
        cfg = session_config()
    except SystemExit as exc:
        await ws.send_json({"type": "log", "level": "error", "text": str(exc)})
        await ws.close()
        return

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
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
