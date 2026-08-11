"""เว็บเซิร์ฟเวอร์: เสิร์ฟหน้าเว็บ + สะพาน WebSocket ไปยัง VoiceChat

รัน:  python3 -m web.server            (เปิด http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vc.config import Config, load_config

from .bridge import Outbox
from .session import WebSession, run_selftest

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

log = logging.getLogger("voicechat.web")

app = FastAPI(title="Voice Link", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def session_config() -> Config:
    """คอนฟิกใหม่ต่อหนึ่ง session (แต่ละแท็บปรับ mute/echo แยกกันได้)

    ค่าเริ่มต้นฝั่งเว็บปิด echo guard ของเราเอง เพราะเบราว์เซอร์มี AEC
    (echoCancellation) ที่ทำงานดีกว่าอยู่แล้ว — เปิดกลับได้จากปุ่มในหน้าเว็บ
    """
    cfg = load_config()
    cfg.echo_guard = False
    return cfg


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def api_config() -> JSONResponse:
    cfg = session_config()
    return JSONResponse({
        "chat_model": cfg.chat_model,
        "asr_model": cfg.asr_model,
        "tts_model": cfg.tts_model,
        "base_url": cfg.base_url,
        "mic_sr": cfg.mic_sr,
        "speaker_sr": cfg.speaker_sr,
        "frame_ms": cfg.frame_ms,
    })


@app.post("/api/selftest")
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
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
    worker = threading.Thread(target=session.run, name="session", daemon=True)
    worker.start()
    pump = asyncio.create_task(_pump(ws, out))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                session.feed_audio(data)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                session.handle_client(payload)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error: %r", exc)
    finally:
        session.request_stop()
        await asyncio.to_thread(worker.join, 5.0)
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
    print(f"\n  เปิดเบราว์เซอร์ที่  http://{args.host}:{args.port}\n")
    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
