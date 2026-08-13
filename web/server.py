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
from vc.tts_edge import THAI_VOICES, voice_label

from .bridge import Outbox
from .session import WebSession, run_selftest

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

log = logging.getLogger("voicechat.web")

app = FastAPI(title="Voice Link", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


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
        items.append({"value": "api", "label": f"เซิร์ฟเวอร์ · {cfg.tts_model}"})
    names = [name for name, _label, _gender in THAI_VOICES]
    if cfg.edge_voice and cfg.edge_voice not in names:
        names.append(cfg.edge_voice)     # เสียงที่ตั้งเองใน .env ต้องเลือกกลับมาได้
    for name in names:
        items.append({"value": f"edge:{name}", "label": f"Edge · {voice_label(name)}"})
    return items


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
        "tts_label": cfg.tts_label,
        "tts_voice": cfg.tts_voice_value,
        "tts_voices": voice_options(cfg),
        "base_url": cfg.base_url,
        "mic_sr": cfg.mic_sr,
        "speaker_sr": cfg.speaker_sr,
        "frame_ms": cfg.frame_ms,
    })


@app.post("/api/selftest")
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, voice: str = "") -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    out = Outbox(loop)

    try:
        cfg = session_config()
    except SystemExit as exc:
        await ws.send_json({"type": "log", "level": "error", "text": str(exc)})
        await ws.close()
        return

    if voice:
        # เบราว์เซอร์ส่งเสียงที่จำไว้มาตั้งแต่ตอนต่อ WebSocket — ต้องตั้งค่า cfg
        # ให้เสร็จก่อน WebSession เริ่มทักทาย ไม่งั้นทักทายไปด้วยเสียง/เพศจาก .env
        # ก่อนที่ฝั่งเว็บจะทันส่งคำสั่งสลับเสียงมาทีหลัง (แข่งกันตอนเริ่ม session)
        cfg.set_tts_voice(voice)   # ค่าผิดรูปถูกเมิน แล้วตกกลับไปใช้ค่าใน .env

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
