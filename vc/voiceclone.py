"""เสียงอ้างอิงสำหรับ voice cloning ของ k2-fsa/OmniVoice

OmniVoice ผ่าน LiteLLM โคลนเสียงได้จริง แต่ต้องส่ง **ทั้งคู่**:

    {"ref_audio": "data:audio/wav;base64,...", "ref_text": "<คำอ่านของไฟล์นั้น>"}

รูปแบบอื่นใช้ไม่ได้ทั้งหมด (ทดสอบกับเซิร์ฟเวอร์จริงแล้ว):

- ส่ง base64 เปล่า ๆ ไม่มี `data:` นำหน้า → 500
- ส่งเป็น dict `{"data": ..., "format": "wav"}` หรือ list → 500
- ส่ง `ref_audio` เดี่ยว ๆ ไม่มี `ref_text` → ตอบ 200 แต่ได้เสียงแทบเงียบ (rms 0.01)
  ซึ่ง ASR ถอดกลับมาไม่เป็นภาษา — พังแบบเงียบ ๆ จึงต้องกันไว้ที่ชั้นนี้
- ชื่อคีย์อื่น (`reference_audio`, `prompt_audio`) ถูก LiteLLM ตัดทิ้งเงียบ ๆ
  ได้ 200 แต่เป็นเสียงสุ่มเหมือนเดิม จึงดูจากสถานะ HTTP อย่างเดียวไม่พอ

ทำไมต้องโคลน: OmniVoice แบบไม่ส่งเสียงอ้างอิงจะสุ่มเสียงคนพูดใหม่ทุก request
คำตอบเดียวที่ถูกหั่นเป็นหลายก้อนจึงกลายเป็นคนละคนพูดกลางประโยค วัดจริงด้วย
ระยะห่าง MFCC ระหว่างก้อน: ไม่ส่งอ้างอิง = 0.77 · ส่งอ้างอิง = 0.98 (1.00 = คนเดียวกัน)

ความดังก็ถูกโคลนมาด้วย ไฟล์อ้างอิงที่อัดมาเบาจะทำให้ AI พูดเบาตามทั้งระบบ
(วัดจริง: ไฟล์อ้างอิง rms 0.029 → เสียงที่สังเคราะห์ได้ rms 0.017 เทียบกับ 0.18
เมื่อไม่โคลน) จึงต้องปรับความดังไฟล์อ้างอิงให้เต็มสเกลก่อนเสมอ
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("voicechat.voiceclone")

REF_PEAK = 0.95        # ปรับไฟล์อ้างอิงให้ดังเกือบเต็มสเกลก่อนส่ง (ดู docstring)
MIN_REF_SEC = 1.0      # สั้นกว่านี้โคลนไม่ติด ได้เสียงเพี้ยน
MAX_REF_SEC = 30.0     # ยาวกว่านี้ payload บวม/ช้าโดยไม่ได้ความเหมือนเพิ่ม


class ReferenceError(RuntimeError):
    """ไฟล์เสียงอ้างอิงใช้ไม่ได้ — ผู้เรียกต้องตกกลับไปพูดแบบไม่โคลน"""


def resolve_path(raw: str) -> Path:
    """พาธสัมพัทธ์ให้อิงรากโปรเจกต์เสมอ ไม่ใช่ cwd ตอนสั่งรัน"""
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def read_wav_mono(data: bytes) -> tuple[np.ndarray, int]:
    """อ่าน WAV 16-bit เป็น float mono ช่วง [-1, 1]"""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ReferenceError(f"อ่านไฟล์ WAV ไม่ได้: {exc}") from exc
    if width != 2:
        raise ReferenceError(f"รองรับเฉพาะ WAV 16-bit (ได้ {width * 8}-bit)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x / 32768.0, sr


def write_wav_mono(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def prepare(data: bytes, max_sec: float = 0.0, normalize: bool = True) -> bytes:
    """ปรับไฟล์อ้างอิงให้พร้อมส่ง: mono → ตัดความยาว → ปรับความดัง

    `max_sec=0` = ไม่ตัด ค่าเริ่มต้นเป็นแบบนี้เพราะ `ref_text` ต้องตรงกับเสียง
    ที่ส่งไปจริง การตัดจึงทำให้ท้ายประโยคใน ref_text ไม่มีเสียงคู่กัน
    (ยังใช้ได้ วัดได้ 0.94 เทียบกับ 0.98 — แลกกับ payload ครึ่งเดียวและเร็วขึ้น
    ~0.7 วิ/ก้อน จึงเปิดให้เลือกเองผ่าน TTS_REF_MAX_SEC)
    """
    x, sr = read_wav_mono(data)
    if x.size == 0:
        raise ReferenceError("ไฟล์เสียงอ้างอิงว่างเปล่า")
    if max_sec > 0:
        x = x[: int(max_sec * sr)]
    dur = x.size / sr
    if dur < MIN_REF_SEC:
        raise ReferenceError(f"เสียงอ้างอิงสั้นเกินไป ({dur:.1f} วิ ต้องอย่างน้อย "
                             f"{MIN_REF_SEC:.0f} วิ)")
    if dur > MAX_REF_SEC:
        x = x[: int(MAX_REF_SEC * sr)]
    if normalize:
        peak = float(np.abs(x).max())
        if peak > 1e-6:
            x = x * (REF_PEAK / peak)
    return write_wav_mono(x, sr)


def to_data_uri(wav: bytes) -> str:
    """เซิร์ฟเวอร์รับเฉพาะรูป data URI — base64 เปล่า ๆ ตอบ 500 (ดู docstring บนสุด)"""
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def sidecar(path: Path) -> Path:
    """ไฟล์คำอ่านที่วางคู่กับไฟล์เสียง เช่น thai_default.wav → thai_default.txt"""
    return path.with_suffix(".txt")


class VoiceReference:
    """โหลด/แปลง/จำไฟล์เสียงอ้างอิงไว้ครั้งเดียว แล้วแจกให้ทุก request ของ TTS

    ถูกเรียกจากเธรด TTS worker จึงต้องมีล็อกของตัวเอง และเมื่อโหลดพลาดต้อง
    "พังครั้งเดียวจำไว้" ไม่ใช่ลองใหม่ทุกก้อนจนพูดตะกุกตะกัก
    """

    def __init__(self, path: str, text: str = "", max_sec: float = 0.0,
                 normalize: bool = True) -> None:
        self.path = resolve_path(path)
        self._text = (text or "").strip()
        self._max_sec = max_sec
        self._normalize = normalize
        self._lock = threading.Lock()
        self._uri: str | None = None
        self._failed: str | None = None

    # -- คำอ่านของไฟล์อ้างอิง -------------------------------------------------
    def _load_text(self, transcribe) -> str:
        """ลำดับที่มา: ค่าใน .env → ไฟล์ .txt ข้าง ๆ → ถอดเสียงเอาเอง (แล้วจำลงไฟล์)

        ที่ต้องมี fallback ถึงชั้น ASR เพราะผู้ใช้ที่เอาไฟล์เสียงตัวเองมาวาง
        มักไม่มีคำอ่าน และถ้าส่ง ref_audio โดยไม่มี ref_text จะได้เสียงเงียบ
        แบบไม่มีข้อความแจ้งเตือน
        """
        if self._text:
            return self._text
        side = sidecar(self.path)
        try:
            cached = side.read_text(encoding="utf-8").strip()
        except OSError:
            cached = ""
        if cached:
            self._text = cached
            return cached
        if transcribe is None:
            raise ReferenceError(
                f"ไม่มีคำอ่านของไฟล์เสียงอ้างอิง — ตั้ง TTS_REF_TEXT ใน .env "
                f"หรือสร้างไฟล์ {side.name} วางคู่กับ {self.path.name}")
        x, sr = read_wav_mono(self.path.read_bytes())
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        text = (transcribe(pcm, sr) or "").strip()
        if not text:
            raise ReferenceError("ถอดคำอ่านของไฟล์เสียงอ้างอิงไม่ได้ (ASR คืนค่าว่าง)")
        try:
            side.write_text(text + "\n", encoding="utf-8")
            log.info("บันทึกคำอ่านของเสียงอ้างอิงไว้ที่ %s", side)
        except OSError as exc:      # เขียนไม่ได้ก็ยังใช้ได้ แค่ต้องถอดใหม่รอบหน้า
            log.warning("เขียน %s ไม่ได้: %s", side, exc)
        self._text = text
        return text

    # -- ค่าที่เอาไปใส่ payload ----------------------------------------------
    def payload(self, transcribe=None) -> dict | None:
        """คืน {"ref_audio": ..., "ref_text": ...} หรือ None เมื่อใช้ไม่ได้

        ไม่โยน exception ออกไป เพราะเสียงอ้างอิงเสียไม่ควรทำให้ AI พูดไม่ได้เลย
        — ตกกลับไปใช้เสียงสุ่มของ OmniVoice แล้วเตือนครั้งเดียวพอ
        """
        with self._lock:
            if self._uri is not None and self._text:
                return {"ref_audio": self._uri, "ref_text": self._text}
            if self._failed is not None:
                return None
            try:
                raw = self.path.read_bytes()
                text = self._load_text(transcribe)
                self._uri = to_data_uri(prepare(raw, self._max_sec, self._normalize))
            except Exception as exc:  # noqa: BLE001 — เสียงอ้างอิงพังต้องไม่ทำให้พูดไม่ได้
                self._failed = str(exc)
                log.warning("ปิดการโคลนเสียง ใช้เสียงสุ่มของ OmniVoice แทน: %s", exc)
                return None
            log.info("โหลดเสียงอ้างอิง %s แล้ว (%.0f KB, คำอ่าน %d ตัวอักษร)",
                     self.path.name, len(self._uri) / 1024, len(text))
            return {"ref_audio": self._uri, "ref_text": text}

    @property
    def error(self) -> str | None:
        return self._failed
