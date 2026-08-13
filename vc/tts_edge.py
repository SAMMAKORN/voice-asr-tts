"""แบ็กเอนด์ TTS ตัวที่สอง: Microsoft Edge Read Aloud (edge-tts)

ทำไมต้องมีตัวเลือกนี้ — OmniVoice ผ่าน vLLM ล็อกเสียงข้ามก้อนไม่ได้
(ดูหมายเหตุใน .env.example) คำตอบยาว ๆ ที่ถูกหั่นเป็นหลายก้อนจึงออกมา
เป็นคนละคนสลับไปมา ส่วน Edge ระบุ voice ได้ตรง ๆ ทุกก้อนจึงเป็นเสียง
เดียวกันตลอดบทสนทนา และไม่ต้องใช้ API key ด้วย

ข้อแลกเปลี่ยน: ต้องต่อเน็ตออกไปหา Microsoft และเสียงที่ได้กลับมาเป็น MP3
จึงต้องถอดรหัสก่อนด้วย soundfile (libsndfile ≥ 1.1 อ่าน MP3 ได้ในตัว
ไม่ต้องติดตั้ง ffmpeg แยก)
"""
from __future__ import annotations

import asyncio
import io
import re
import time

import numpy as np

from .api import ApiError, resample_i16

# เสียงภาษาไทยที่ Edge มีให้ (ตรวจจาก edge-tts list-voices เมื่อ 2026-08-13)
THAI_VOICES: tuple[tuple[str, str, str], ...] = (
    ("th-TH-PremwadeeNeural", "เปรมวดี · หญิง", "female"),
    ("th-TH-NiwatNeural", "นิวัฒน์ · ชาย", "male"),
)
DEFAULT_VOICE = THAI_VOICES[0][0]

_PERCENT = re.compile(r"^[+-]\d+%$")
_HERTZ = re.compile(r"^[+-]\d+Hz$")


def voice_label(voice: str) -> str:
    """ชื่อที่อ่านง่ายสำหรับโชว์บน UI — เสียงนอกลิสต์ก็คืนชื่อดิบไป"""
    for name, label, _gender in THAI_VOICES:
        if name == voice:
            return label
    return voice


def voice_gender(voice: str) -> str:
    """เพศของเสียง ("male"/"female") ใช้เลือกคำลงท้ายให้ตรงเสียงจริง

    เสียงนอกลิสต์ (ตั้งเองใน .env) ถือเป็นชาย — พฤติกรรมเดิมของโปรแกรมก่อนมีฟีเจอร์นี้
    """
    for name, _label, gender in THAI_VOICES:
        if name == voice:
            return gender
    return "male"


def _norm(value: str, pattern: re.Pattern[str], default: str) -> str:
    """edge-tts ตรวจรูปแบบเข้มมาก ('+10%' ผ่าน แต่ '10%' โยน error ทิ้ง)

    ค่าที่ผู้ใช้พิมพ์ใน .env ผิดรูปนิดหน่อยจึงเติม + ให้ก่อน แล้วค่อยยอมแพ้
    """
    v = (value or "").strip().replace(" ", "")
    if not v:
        return default
    if not v[0] in "+-":
        v = "+" + v
    return v if pattern.match(v) else default


async def _fetch_mp3(text: str, voice: str, rate: str,
                     volume: str, pitch: str) -> bytes:
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return bytes(buf)


def _decode(mp3: bytes, dst_sr: int) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(mp3), dtype="int16", always_2d=True)
    mono = pcm[:, 0] if pcm.shape[1] == 1 else pcm.mean(axis=1).astype(np.int16)
    return resample_i16(np.ascontiguousarray(mono), sr, dst_sr)


def synthesize(text: str, dst_sr: int, voice: str = DEFAULT_VOICE,
               rate: str = "+0%", volume: str = "+0%",
               pitch: str = "+0Hz") -> np.ndarray:
    """สังเคราะห์เสียงหนึ่งก้อน คืน PCM 16-bit โมโนที่ dst_sr

    ถูกเรียกจากเธรด TTS (ไม่มี event loop ของตัวเอง) จึงใช้ asyncio.run ได้ตรง ๆ
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.int16)

    try:
        import edge_tts  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise ApiError(
            "TTS_BACKEND=edge ต้องติดตั้งเพิ่ม: pip install edge-tts soundfile "
            f"({exc})"
        ) from exc

    rate = _norm(rate, _PERCENT, "+0%")
    volume = _norm(volume, _PERCENT, "+0%")
    pitch = _norm(pitch, _HERTZ, "+0Hz")

    # เชื่อมต่อออกเน็ตทุกครั้ง หลุดเป็นครั้งคราวเป็นเรื่องปกติ — ลองซ้ำหนึ่งรอบ
    # ดีกว่าปล่อยให้ประโยคนั้นเงียบหายไปทั้งก้อน
    last: Exception | None = None
    for attempt in range(2):
        try:
            mp3 = asyncio.run(_fetch_mp3(text, voice, rate, volume, pitch))
            if not mp3:
                raise RuntimeError("ไม่ได้รับข้อมูลเสียงกลับมา")
            return _decode(mp3, dst_sr)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == 0:
                time.sleep(0.4)

    raise ApiError(f"edge-tts ({voice}) ล้มเหลว: {last}")
