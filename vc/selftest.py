"""ตรวจว่าต่อ API ได้จริงครบวง TTS → ASR → LLM โดยไม่ใช้ไมโครโฟน

เดิมมีสองชุดที่เขียนแยกกัน (`voice_chat.py` สำหรับ CLI และ `web/session.py`
สำหรับหน้าเว็บ) แล้ว**พฤติกรรมต่างกันไปแล้ว** เช่นฝั่งเว็บถือว่า ASR ที่ไม่ได้
ข้อความคือ "ไม่ผ่าน" แต่ฝั่ง CLI ไม่ตรวจเลย และฝั่งเว็บหยุดสตรีมที่ 400 ตัวอักษร
แต่ฝั่ง CLI อ่านจนจบ (P3-21 ข้อ 4)

ตอนนี้เหลือฟังก์ชันเดียวที่คืน "ผลลัพธ์เป็นโครงสร้าง" แล้วผู้เรียกแต่ละฝ่าย
รับไปวาดเอง — CLI พิมพ์ลงเทอร์มินัล ส่วนหน้าเว็บส่งเป็น JSON
"""
from __future__ import annotations

import threading
import time

from .api import ApiClient, ApiError
from .chat import GREETING          # ข้อความทักทายมีที่มาที่เดียว
from .config import Config

SAMPLE = "สวัสดีครับ วันนี้อากาศที่กรุงเทพเป็นอย่างไรบ้าง"
# อ่านคำตอบพอเป็นตัวอย่าง ไม่ต้องรอให้โมเดลพูดจบ (ประหยัดเวลาตอนตรวจระบบ)
MAX_REPLY_CHARS = 400


def run_selftest(cfg: Config) -> dict:
    """คืน `{"ok": bool, "steps": [...], "greeting": str}` — ไม่พิมพ์อะไรออกเอง"""
    api = ApiClient(cfg)
    steps: list[dict] = []

    def add(name: str, model: str, ok: bool, detail: str, ms: int) -> None:
        steps.append({"name": name, "model": model, "ok": ok,
                      "detail": detail, "ms": ms})

    try:
        t0 = time.perf_counter()
        pcm = api.synthesize(SAMPLE, cfg.mic_sr)
        add("TTS", cfg.tts_model, True,
            f"สังเคราะห์เสียงได้ {pcm.size / cfg.mic_sr:.2f} วินาที",
            int((time.perf_counter() - t0) * 1000))

        t0 = time.perf_counter()
        text = api.transcribe(pcm, cfg.mic_sr)
        add("ASR", cfg.asr_model, bool(text), text or "ถอดเสียงไม่ได้ข้อความ",
            int((time.perf_counter() - t0) * 1000))

        t0 = time.perf_counter()
        cancel = threading.Event()
        msgs = [{"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": text or SAMPLE}]
        out, first = "", None
        for delta in api.chat_stream(msgs, cancel):
            if first is None:
                first = int((time.perf_counter() - t0) * 1000)
            out += delta
            if len(out) > MAX_REPLY_CHARS:
                cancel.set()
        add("LLM", cfg.chat_model, first is not None,
            out.strip()[:160] or "โมเดลไม่ส่งข้อความกลับมา",
            first or int((time.perf_counter() - t0) * 1000))
    except ApiError as exc:
        steps.append({"name": "ผิดพลาด", "model": "", "ok": False,
                      "detail": str(exc)[:300], "ms": 0})
    except Exception as exc:  # noqa: BLE001
        steps.append({"name": "ผิดพลาด", "model": "", "ok": False,
                      "detail": repr(exc)[:300], "ms": 0})
    finally:
        api.close()

    return {"ok": bool(steps) and all(s["ok"] for s in steps), "steps": steps,
            "greeting": GREETING}
