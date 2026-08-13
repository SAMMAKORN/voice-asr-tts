#!/usr/bin/env python3
"""ทดสอบเว็บแอปครบวงจรโดยไม่ต้องเปิดเบราว์เซอร์

เปิดเซิร์ฟเวอร์จริง แล้วต่อ WebSocket เข้าไปทำตัวเป็นเบราว์เซอร์:
ป้อนเสียงพูดจริง (สร้างจาก TTS) → ดูว่าถอดเสียง เรียก LLM แล้วส่งเสียงกลับมา
จากนั้นลองพูดแทรกกลางคัน แล้วดูว่าระบบสั่งหยุดเสียงจริง

  python3 tests/test_web_e2e.py        (ต้องต่อ API ได้ ใช้เวลาราวครึ่งนาที)
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn                                       # noqa: E402
import websockets                                    # noqa: E402

from vc.api import ApiClient, resample_i16           # noqa: E402
from vc.config import load_config                    # noqa: E402

PASS, FAIL = 0, 0
FRAME_MS = 20
MIC_SR = 16000

# ตั้ง token ให้แน่นอนก่อนเซิร์ฟเวอร์ถูก import (P1-1 — WebSocket ต้องมี token)
AUTH_TOKEN = "e2e-test-token-0123456789"
os.environ["WEB_AUTH_TOKEN"] = AUTH_TOKEN


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✓ {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """รัน uvicorn ในเธรดแยกเพื่อให้ทดสอบจบในตัว"""

    def __init__(self, port: int):
        cfg = uvicorn.Config("web.server:app", host="127.0.0.1", port=port,
                             log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "Server":
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.1)
        raise RuntimeError("เซิร์ฟเวอร์ไม่ขึ้นภายในเวลาที่กำหนด")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


class FakeBrowser:
    """จำลองฝั่งเบราว์เซอร์: สตรีมไมค์ขึ้นไป และเก็บทุกอย่างที่ส่งลงมา"""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []
        self.audio: list[tuple[int, int, int]] = []   # (epoch, seq, จำนวนตัวอย่าง)
        self.captions: dict[str, str] = {"user": "", "assistant": ""}
        self._role = ""
        self.ends: list[dict] = []

    async def reader(self) -> None:
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    e = int(np.frombuffer(msg[0:4], "<u4")[0])
                    s = int(np.frombuffer(msg[4:8], "<u4")[0])
                    self.audio.append((e, s, (len(msg) - 8) // 2))
                    continue
                m = json.loads(msg)
                self.events.append(m)
                if m["type"] == "begin":
                    self._role = m["role"]
                elif m["type"] == "delta" and self._role:
                    self.captions[self._role] += m["text"]
                elif m["type"] == "end":
                    self.ends.append(m)
                    self._role = ""
        except websockets.exceptions.ConnectionClosed:
            pass

    async def feed(self, pcm: np.ndarray, speed: float = 2.0) -> None:
        """ส่งเสียงเป็นเฟรม 20 ms เหมือน AudioWorklet ของจริง"""
        n = int(MIC_SR * FRAME_MS / 1000)
        for i in range(0, pcm.size - n + 1, n):
            await self.ws.send(pcm[i:i + n].astype("<i2").tobytes())
            await asyncio.sleep(FRAME_MS / 1000 / speed)

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.events]

    def first(self, kind: str) -> dict | None:
        return next((e for e in self.events if e["type"] == kind), None)


def silence(ms: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.normal(0, 0.0008, int(MIC_SR * ms / 1000)) * 32767).astype(np.int16)


def make_speech(api: ApiClient, text: str) -> np.ndarray:
    """สร้างเสียงพูดจริงด้วย TTS เพื่อใช้เป็นอินพุตของไมค์จำลอง"""
    pcm = api.synthesize(text, MIC_SR)
    peak = int(np.abs(pcm).max()) or 1
    return (pcm.astype(np.float32) * (0.35 * 32767 / peak)).astype(np.int16)


async def scenario(port: int, speech: np.ndarray, barge: np.ndarray) -> FakeBrowser:
    # ตั้งแต่ P1-1 เซิร์ฟเวอร์ต้องการ token (client นี้ไม่มี Origin เหมือนเบราว์เซอร์)
    url = f"ws://127.0.0.1:{port}/ws?token={AUTH_TOKEN}"
    async with websockets.connect(url, max_size=8 << 20) as ws:
        br = FakeBrowser(ws)
        task = asyncio.create_task(br.reader())

        # 1) เงียบให้ระบบวัดเสียงรบกวน (เหมือนตอนเปิดหน้าเว็บครั้งแรก)
        await br.feed(silence(1600), speed=4.0)
        for _ in range(60):
            if br.first("ready"):
                break
            await asyncio.sleep(0.1)

        # 2) รอให้ทักทายจบก่อน แล้วค่อยพูด
        t0 = time.time()
        while time.time() - t0 < 12 and not br.audio:
            await br.feed(silence(200), speed=4.0)
        for e, s, n in list(br.audio):          # บอกว่าเล่นเสียงทักทายจบแล้ว
            await ws.send(json.dumps({"type": "played", "epoch": e, "seq": s}))
        greet_chunks = len(br.audio)

        # 3) พูดใส่ไมค์จริง
        await br.feed(speech)
        await br.feed(silence(900))

        # 4) รอคำตอบ (ข้อความ + เสียง)
        t0 = time.time()
        while time.time() - t0 < 60:
            if len(br.audio) > greet_chunks and br.captions["assistant"]:
                break
            await br.feed(silence(200), speed=4.0)

        replied = len(br.audio)

        # 5) พูดแทรกกลางที่ AI กำลังพูด (ยังไม่ส่ง played จึงยังถือว่าพูดอยู่)
        await br.feed(barge)
        await br.feed(silence(700))
        t0 = time.time()
        while time.time() - t0 < 20 and not br.first("stop"):
            await br.feed(silence(200), speed=4.0)

        br.greet_chunks = greet_chunks          # type: ignore[attr-defined]
        br.replied = replied                    # type: ignore[attr-defined]
        await ws.close()
        await asyncio.wait_for(task, timeout=5)
        return br


def main() -> int:
    cfg = load_config()
    print("กำลังเตรียมเสียงพูดสำหรับทดสอบ (เรียก TTS จริง)...")
    api = ApiClient(cfg)
    try:
        speech = make_speech(api, "ช่วยเล่าเรื่องดาวอังคารให้ฟังหน่อยครับ")
        barge = make_speech(api, "เดี๋ยวก่อนครับ พอแค่นี้ก่อน")
    finally:
        api.close()
    print(f"  เสียงคำถาม {speech.size / MIC_SR:.1f}s · เสียงพูดแทรก {barge.size / MIC_SR:.1f}s")

    port = free_port()
    with Server(port):
        br = asyncio.run(scenario(port, speech, barge))

    print("\n[1] เชื่อมต่อและตั้งต้นระบบ")
    ready = br.first("ready")
    check("เซิร์ฟเวอร์ส่งค่าตั้งต้นกลับมา", ready is not None)
    if ready:
        check("บอกชื่อโมเดลครบ",
              all(ready.get(k) for k in ("chat_model", "asr_model", "tts_model")),
              f"{ready.get('chat_model')} / {ready.get('asr_model')}")
    cal = br.first("calibrated")
    check("วัดเสียงรบกวนจากเสียงที่เบราว์เซอร์ส่งมา", cal is not None,
          f"noise={cal.get('noise') if cal else '-'}")
    check("ทักทายด้วยเสียงตอนเริ่ม", br.greet_chunks > 0,
          f"{br.greet_chunks} ก้อนเสียง")

    print("\n[2] พูดใส่ไมค์แล้วได้คำตอบครบวง")
    check("ถอดเสียงที่พูดออกมาเป็นข้อความได้", len(br.captions["user"]) > 0,
          repr(br.captions["user"][:60]))
    check("LLM ตอบกลับเป็นข้อความ", len(br.captions["assistant"]) > 0,
          repr(br.captions["assistant"][:60]))
    check("ส่งเสียงคำตอบกลับมาเล่น", br.replied > br.greet_chunks,
          f"{br.replied - br.greet_chunks} ก้อน")
    asr = next((e for e in br.events
                if e["type"] == "metric" and e.get("kind") == "asr"), None)
    check("รายงาน latency ของ ASR ขึ้นหน้าเว็บ", asr is not None,
          f"{asr.get('latency_ms')} ms" if asr else "")

    print("\n[3] พูดแทรกกลางประโยค")
    check("ตรวจพบว่าผู้ใช้เริ่มพูดขณะ AI พูดอยู่",
          any(e["type"] == "speech" and e.get("state") == "start"
              for e in br.events))
    check("สั่งให้เบราว์เซอร์หยุดเล่นเสียงทันที", br.first("stop") is not None)
    check("บันทึกว่าคำตอบถูกพูดขัด",
          any(e.get("interrupted") for e in br.ends),
          f"ends={[e.get('interrupted') for e in br.ends]}")

    print()
    if FAIL:
        print(f"ล้มเหลว {FAIL} ข้อ (ผ่าน {PASS}) ✗")
        return 1
    print(f"ผ่านทั้งหมด {PASS} ข้อ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
