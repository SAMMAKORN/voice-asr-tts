"""โหลดค่าตั้งต้นทั้งหมดจาก .env พร้อมตรวจช่วงค่าตอนเริ่มโปรแกรม (P3-19)

กติกาการตรวจ
* ค่าที่ "ไม่ใช่ตัวเลข" = ตั้งค่าผิดชัด ๆ → หยุดทันทีพร้อมบอกชื่อ key ค่าที่ได้รับ
  และช่วงที่ยอมรับ (เดิมกลืนเงียบแล้วใช้ค่าปริยาย ผู้ใช้ไม่รู้ว่าที่ตั้งไปไม่มีผล)
* ค่าที่ "อยู่นอกช่วง" = เจตนาชัดแต่เกินขอบ → บีบเข้าช่วง (clamp) พร้อม WARNING
  เพราะการหยุดโปรแกรมด้วยเรื่องแบบนี้ทำให้ใช้งานไม่ได้เลยโดยไม่จำเป็น
* คู่ค่าที่รวมกันแล้วเป็นปัญหา (เช่นสังเคราะห์เสียงทีเดียวด้วยข้อความยาวมาก
  = ผู้ใช้รอเงียบ 16 วินาที ตามที่เจอในบันทึกจริง) → WARNING บอกผลกระทบตรง ๆ
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# รูปแบบชื่อเสียงของ Microsoft เช่น th-TH-PremwadeeNeural
VOICE_NAME = re.compile(r"^[A-Za-z]{2,3}-[A-Za-z0-9]{2,8}-[A-Za-z0-9]{2,40}$")

# เวลาของระบบ (P3-20) — บอกโมเดลว่า "ตามเวลาประเทศไทย" จึงต้องเป็นเวลาไทยจริง
# ไม่ใช่เวลาของเครื่อง (บน container ที่ตั้ง TZ=UTC เดิมเพี้ยนไป 7 ชั่วโมง)
DEFAULT_TZ = "Asia/Bangkok"

THAI_DAYS = ("วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี",
             "วันศุกร์", "วันเสาร์", "วันอาทิตย์")
THAI_MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม")

# ช่วงที่ยอมรับของทุกค่าตัวเลขใน .env — (ต่ำสุด, สูงสุด, คำอธิบายภาษาไทย)
# ช่วงตั้งให้กว้างพอสำหรับการจูนจริง แต่แคบพอจะจับ "พิมพ์ผิดหนึ่งหลัก" ได้
RANGES: dict[str, tuple[float, float, str]] = {
    "CHAT_TEMPERATURE": (0.0, 2.0, "ความสร้างสรรค์ของโมเดล"),
    "CHAT_MAX_TOKENS": (16, 8192, "token สูงสุดต่อคำตอบ"),
    "HTTP_RETRY_MAX": (0, 10, "จำนวนครั้งที่ลองใหม่"),
    "HTTP_RETRY_BASE_MS": (10, 10_000, "ฐานเวลาคอยของ backoff"),
    "REPLY_MAX_SENTENCES": (0, 100, "เพดานจำนวนประโยค (0 = ปิด)"),
    "REPLY_MAX_CHARS": (20, 100_000, "เพดานตัวอักษรต่อคำตอบ"),
    "KEEP_FINDINGS": (0, 50, "ผลค้นเว็บย้อนหลังที่จำไว้"),
    "MIC_GAIN": (0.0, 20.0, "อัตราขยายเสียงไมค์ (0 = อัตโนมัติ)"),
    "MIC_TARGET_NOISE": (0.0001, 0.5, "ระดับเสียงรบกวนเป้าหมาย"),
    "VAD_ABS_THRESHOLD": (0.0001, 1.0, "เกณฑ์ระดับเสียงที่นับว่าพูด"),
    "VAD_NOISE_MULT": (1.0, 20.0, "ตัวคูณเหนือเสียงรบกวน"),
    "VAD_CONFIRM_MS": (0, 5_000, "มิลลิวินาทีที่ต้องพูดต่อเนื่อง"),
    "VAD_CONFIRM_MS_PLAYBACK": (0, 5_000, "เท่าข้างบน แต่ตอน AI พูด"),
    "VAD_END_SILENCE_MS": (100, 5_000, "เงียบเท่าไรถือว่าพูดจบ"),
    "VAD_MIN_UTTERANCE_MS": (50, 10_000, "ความยาวคำพูดต่ำสุด"),
    "VAD_MAX_UTTERANCE_MS": (1_000, 120_000, "ความยาวคำพูดสูงสุด"),
    "ECHO_MARGIN": (1.0, 20.0, "เท่าของเสียงลำโพงที่รั่วเข้าไมค์"),
    "BARGE_IN_MIN_CHARS": (0, 100, "ตัวอักษรขั้นต่ำที่นับว่าพูดแทรกจริง"),
    "TTS_MAX_CHARS": (40, 2_000, "ตัวอักษรต่อหนึ่งคำขอ TTS"),
    "TTS_FIRST_CHARS": (4, 2_000, "ขนาดก้อนแรก"),
    "TTS_CHUNK_CHARS": (8, 2_000, "ขนาดก้อนที่สอง"),
    "TTS_CHUNK_GROWTH": (1.0, 5.0, "อัตราโตของก้อนถัดไป"),
    "TTS_REF_MAX_SEC": (0.0, 30.0, "วินาทีของเสียงอ้างอิงที่ส่งไปโคลน"),
    "SEARCH_RESULTS": (1, 20, "จำนวนผลค้นที่ส่งให้โมเดล"),
    "SEARCH_TIMEOUT": (1.0, 120.0, "วินาทีที่ยอมรอผลค้นทั้งกระบวนการ"),
    "FETCH_MAX_CHARS": (200, 200_000, "ตัวอักษรจากหน้าเว็บที่ส่งให้โมเดล"),
    "FETCH_MAX_BYTES": (10_000, 50_000_000, "ไบต์ที่ยอมดาวน์โหลดต่อหน้า"),
    "TOOL_ROUNDS": (0, 10, "รอบการเรียกเครื่องมือต่อคำถาม"),
    "LOG_RETENTION_DAYS": (0, 3_650, "ลบบันทึกที่เก่ากว่ากี่วัน (0 = ไม่ลบ)"),
}

# เตือนซ้ำข้อความเดิมครั้งเดียวต่อ process — โหมดเว็บเรียก load_config() ทุก session
_warned: set[str] = set()


def warn(message: str, sink: list[str] | None = None) -> None:
    """เตือนไปที่ stderr (ผู้ใช้เห็นทันทีตอนเปิด) และเก็บไว้ให้ตรวจย้อนได้"""
    if sink is not None:
        sink.append(message)
    if message in _warned:
        return
    _warned.add(message)
    print(f"  ⚠️  WARNING: {message}", file=sys.stderr)


def _num(key: str, default: float, warnings: list[str] | None = None) -> float:
    """อ่านค่าตัวเลขหนึ่งค่าพร้อมตรวจชนิดและช่วง"""
    lo, hi, what = RANGES.get(key, (-math.inf, math.inf, ""))
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isnan(value) or math.isinf(value):
        raise SystemExit(
            f"ค่า {key} ใน .env ต้องเป็นตัวเลข แต่ได้ {raw!r}\n"
            f"    {what} — ช่วงที่ยอมรับ {_fmt(lo)} ถึง {_fmt(hi)} "
            f"(ค่าเริ่มต้น {_fmt(default)})")
    if value < lo or value > hi:
        clamped = min(max(value, lo), hi)
        warn(f"{key}={raw} อยู่นอกช่วง {_fmt(lo)}-{_fmt(hi)} ({what}) "
             f"— ใช้ค่า {_fmt(clamped)} แทน", warnings)
        value = clamped
    return value


def _fmt(value: float) -> str:
    if value in (math.inf, -math.inf):
        return "ไม่จำกัด"
    return str(int(value)) if float(value).is_integer() else str(value)


def _f(key: str, default: float, warnings: list[str] | None = None) -> float:
    return _num(key, default, warnings)


def _i(key: str, default: int, warnings: list[str] | None = None) -> int:
    return int(_num(key, default, warnings))


def zone(name: str | None = None) -> ZoneInfo:
    """timezone ที่ใช้ทั้งระบบ — ตั้งได้ด้วย `APP_TZ` (ค่าเริ่มต้น Asia/Bangkok)

    ชื่อโซนที่ไม่รู้จักไม่ควรทำให้เปิดโปรแกรมไม่ได้ จึงเตือนแล้วถอยไปใช้เวลาไทย
    """
    key = (name or os.environ.get("APP_TZ") or "").strip() or DEFAULT_TZ
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        warn(f"APP_TZ={key!r} ไม่ใช่ชื่อ timezone ที่รู้จัก "
             f"(เช่น Asia/Bangkok, UTC) — ใช้ {DEFAULT_TZ} แทน")
        return ZoneInfo(DEFAULT_TZ)


def now(tz: str | ZoneInfo | None = None) -> datetime:
    """เวลาปัจจุบันที่มี timezone กำกับเสมอ — ห้ามใช้ `datetime.now()` เปล่าในโปรเจกต์นี้"""
    return datetime.now(tz if isinstance(tz, ZoneInfo) else zone(tz))


def log_dir() -> Path:
    """ที่เก็บบันทึกการสนทนา — ตั้งได้ด้วย `LOG_DIR` (P3-18)

    รับได้ทั้ง path สัมบูรณ์และ path สัมพัทธ์ (อ้างจากรากโปรเจกต์ ไม่ใช่ cwd
    เพราะโหมดเว็บถูกสั่งรันจากที่ไหนก็ได้ แต่บันทึกควรไปกองที่เดียวเสมอ)
    """
    raw = (os.environ.get("LOG_DIR") or "").strip()
    if not raw:
        return ROOT / "logs"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (ROOT / path)


TRUE_WORDS = ("1", "true", "yes", "on")
FALSE_WORDS = ("0", "false", "no", "off")


def _b(key: str, default: bool, warnings: list[str] | None = None) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    if not v:
        return default
    if v in TRUE_WORDS:
        return True
    if v in FALSE_WORDS:
        return False
    warn(f"{key}={v!r} ไม่ใช่ค่าเปิด/ปิด (ใช้ได้: {', '.join(TRUE_WORDS)} "
         f"หรือ {', '.join(FALSE_WORDS)}) — ใช้ค่าเริ่มต้น "
         f"{'เปิด' if default else 'ปิด'} แทน", warnings)
    return default


DEFAULT_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วย AI ที่คุยด้วยเสียงกับผู้ใช้เป็นภาษาไทยเป็นหลัก\n"
    "กติกาการตอบ (สำคัญมาก เพราะคำตอบจะถูกอ่านออกเสียง):\n"
    "1. ตอบสั้นที่สุดเท่าที่ตอบได้จริง ปกติ 1-3 ประโยค ห้ามเกิน 4 ประโยคเด็ดขาด\n"
    "   ถ้าเรื่องยาว ให้สรุปหัวใจสำคัญก่อนแล้วถามว่าอยากฟังรายละเอียดต่อไหม\n"
    "2. ห้ามใช้ Markdown, bullet, หัวข้อย่อย, ตาราง, การขึ้นข้อ 1. 2. 3., emoji\n"
    "   หรือสัญลักษณ์พิเศษใด ๆ เพราะอ่านออกเสียงไม่ได้ ให้เขียนติดกันเป็นภาษาพูด\n"
    "3. ห้ามสรุปซ้ำสิ่งที่เพิ่งพูดไปแล้วในคำตอบเดียวกัน พูดครั้งเดียวพอ\n"
    "4. เขียนตัวเลข หน่วย และคำย่อในรูปที่อ่านออกเสียงได้เลย เช่น 'ประมาณ 25 องศา'\n"
    "5. ถ้าคำถามกำกวมหรือข้อความที่ถอดเสียงมาดูเพี้ยน ให้ถามกลับสั้น ๆ เพื่อยืนยัน\n"
    "6. ถ้าถูกผู้ใช้พูดขัดกลางประโยค ให้หยุดเรื่องเดิมทันทีแล้วตอบเรื่องใหม่ที่ผู้ใช้พูด\n"
    "7. ตอบเป็นภาษาไทยเสมอ ยกเว้นผู้ใช้พูดภาษาอื่นหรือขอให้ตอบภาษาอื่น"
)


TOOL_INSTRUCTION = (
    "สำคัญที่สุด: คุณต่ออินเทอร์เน็ตได้ผ่านเครื่องมือ web_search\n"
    "ห้ามตอบเรื่องต่อไปนี้จากความจำของคุณเด็ดขาด ต้องเรียก web_search ก่อนเสมอ — "
    "ราคาทุกชนิด อัตราแลกเปลี่ยน สภาพอากาศ ข่าว ผลกีฬา ตารางเวลา "
    "หรืออะไรก็ตามที่มีคำว่าวันนี้ ตอนนี้ ล่าสุด ปัจจุบัน\n"
    "ความรู้ในตัวคุณเก่าแล้ว การเดาตัวเลขเองถือว่าผิดร้ายแรง "
    "ส่วนคำถามความรู้ทั่วไปที่ไม่เปลี่ยนตามเวลา ตอบเองได้เลยไม่ต้องค้น\n"
    "เมื่อตอบจากผลค้นหา ให้บอกแหล่งที่มาสั้น ๆ แบบภาษาพูด "
    "เช่น 'อ้างอิงจากสมาคมค้าทองคำนะ' พร้อมบอกวันที่ของข้อมูลถ้ามี "
    "ห้ามอ่าน URL ออกเสียง"
)


def _persona_instruction(gender: str) -> str:
    """บอกโมเดลให้แทนตัวเอง/ลงท้ายประโยคให้ตรงเพศของเสียงพูดที่เลือกไว้

    ไม่งั้นโมเดลชอบเดาเป็น 'ผม...ครับ' โดยอัตโนมัติ ทำให้เสียงผู้หญิงพูดคำลงท้ายผิดเพศ
    """
    if gender == "female":
        return ("คุณแทนตัวเองว่า 'ดิฉัน' และลงท้ายประโยคด้วย 'ค่ะ' "
                "(หรือ 'คะ' เมื่อประโยคเป็นคำถามหรือคำขอ) ให้ตรงกับเสียงพูดที่เลือกไว้เสมอ")
    return ("คุณแทนตัวเองว่า 'ผม' และลงท้ายประโยคด้วย 'ครับ' "
            "ให้ตรงกับเสียงพูดที่เลือกไว้เสมอ")


@dataclass
class Config:
    # --- API ---
    base_url: str = ""
    api_key: str = ""
    chat_model: str = "claude-sonnet-5"
    asr_model: str = ""
    tts_model: str = ""
    temperature: float = 0.6
    max_tokens: int = 350
    # ลองใหม่กี่ครั้งเมื่อเน็ตกระตุก/เจอ 429/5xx (ไม่รวมครั้งแรก) — P2-6
    http_retry_max: int = 2
    http_retry_base_ms: int = 300

    # --- เสียง ---
    mic_sr: int = 16000          # อัตราสุ่มที่ ASR ต้องการ
    speaker_sr: int = 24000      # อัตราสุ่มที่ OmniVoice คืนมา
    frame_ms: int = 20
    input_device: int | None = None
    output_device: int | None = None
    mic_gain: float = 0.0        # 0 = คำนวณอัตโนมัติจากเสียงรบกวนที่วัดได้
    mic_target_noise: float = 0.0035   # ระดับเสียงรบกวนที่ถือว่า "gain กำลังดี"

    # --- VAD / barge-in ---
    vad_abs_threshold: float = 0.012
    vad_noise_mult: float = 3.2
    vad_confirm_ms: int = 120
    vad_confirm_ms_playback: int = 340
    vad_end_silence_ms: int = 650
    vad_min_utterance_ms: int = 350
    vad_max_utterance_ms: int = 25000
    echo_guard: bool = True
    echo_margin: float = 3.5
    barge_in_min_chars: int = 2   # ถอดเสียงได้สั้นกว่านี้ = ไม่นับว่าพูดขัดจริง

    # --- พฤติกรรม ---
    lang_hint: str = "th"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tts_enabled: bool = True
    tts_single_request: bool = False  # True = รอคำตอบจบแล้วยิงครั้งเดียว (เสียงคนเดียว)
    tts_max_chars: int = 260          # เพดานต่อหนึ่ง request — ยาวกว่านี้เวลาสังเคราะห์พุ่ง
    tts_first_chars: int = 24         # ก้อนแรกสั้น ๆ เพื่อให้เริ่มพูดเร็วที่สุด
    tts_chunk_chars: int = 60         # ขนาดก้อนที่สอง (ต้องสังเคราะห์ทันก่อนก้อนแรกเล่นจบ)
    tts_chunk_growth: float = 1.8     # ก้อนถัด ๆ ไปโตขึ้นเท่านี้ (ลดจำนวนครั้งที่เสียงเปลี่ยน)
    tts_search_filler: bool = True    # พูด "ขอค้นข้อมูลสักครู่" ระหว่างค้นเน็ตไหม
    # แปลงตัวเลขเป็นตัวหนังสือไทยก่อนสังเคราะห์เสียง เพราะ OmniVoice อ่านตัวเลขไม่ออก
    # (มีผลกับเสียงเท่านั้น หน้าจอ/ประวัติสนทนายังเป็นตัวเลขปกติ — ดู vc/thainum.py)
    tts_read_numbers: bool = True

    # --- โคลนเสียง (เฉพาะแบ็กเอนด์ api / k2-fsa/OmniVoice) ---
    # ไม่ส่งเสียงอ้างอิง = OmniVoice สุ่มเสียงคนพูดใหม่ทุก request คำตอบเดียวที่ถูก
    # หั่นหลายก้อนจึงเปลี่ยนคนกลางประโยค · ส่งอ้างอิง = เสียงเดียวกันทั้งบท (ดู vc/voiceclone.py)
    tts_ref_audio: str = "sound/thai_default.wav"
    tts_ref_text: str = ""            # ว่าง = อ่านจากไฟล์ .txt ข้าง ๆ แล้วค่อยถอดเสียงเอง
    tts_ref_max_sec: float = 0.0      # 0 = ส่งทั้งไฟล์ (ref_text ตรงกับเสียงครบ)
    tts_ref_normalize: bool = True    # ปรับความดังไฟล์อ้างอิงก่อนส่ง — ไฟล์เบาทำให้ AI พูดเบา
    tts_ref_gender: str = "female"    # เพศของเสียงอ้างอิง — ตัดสินคำลงท้าย ครับ/ค่ะ

    # --- แบ็กเอนด์เสียงพูด: "api" = TTS_MODEL บนเซิร์ฟเวอร์ · "edge" = Microsoft Edge ---
    tts_backend: str = "api"
    edge_voice: str = "th-TH-PremwadeeNeural"
    edge_rate: str = "+0%"            # ความเร็ว เช่น +15% เร็วขึ้น, -10% ช้าลง
    edge_volume: str = "+0%"
    edge_pitch: str = "+0Hz"          # ระดับเสียงสูงต่ำ เช่น +20Hz

    # เพดานความยาวคำตอบ บังคับในโค้ด ไม่พึ่ง prompt อย่างเดียว (P2-15)
    # ปิดเพดานตัวอักษร: ตั้งค่าสูงมาก (เช่น 9999) · ปิดเพดานประโยค: ตั้ง 0
    reply_max_sentences: int = 4
    reply_max_chars: int = 320
    save_audio: bool = False
    history_turns: int = 20      # จำนวนข้อความย้อนหลังที่ส่งให้โมเดล
    keep_findings: int = 4       # จำนวนผลค้นเว็บย้อนหลังที่คงไว้ในความจำ

    # เวลาที่ใช้ทั้งการบอกโมเดลและการตั้งชื่อ/ปั๊มเวลาในบันทึก (P3-20)
    tz: str = DEFAULT_TZ

    # --- บันทึกการสนทนา (P3-18) ---
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    log_transcript: bool = True   # False = ไม่เขียนคำพูดลงดิสก์เลย (เก็บแค่ latency)
    log_retention_days: int = 0   # 0 = ไม่ลบอัตโนมัติ · N = ลบ session ที่เก่ากว่า N วัน

    # --- ค้นข้อมูลจากอินเทอร์เน็ต ---
    web_search: bool = True
    search_results: int = 5
    search_timeout: float = 12.0
    fetch_max_chars: int = 3000
    fetch_max_bytes: int = 512_000   # เพดานไบต์ที่ยอมดาวน์โหลดต่อหนึ่งหน้าเว็บ
    tool_rounds: int = 2         # จำนวนรอบสูงสุดที่ยอมให้เรียกเครื่องมือ

    # คำเตือนที่เกิดตอนโหลดค่า (P3-19) — เก็บไว้ให้ UI/เทสต์อ่านได้ นอกจากพิมพ์ออก stderr
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_samples(self) -> int:
        return int(self.mic_sr * self.frame_ms / 1000)

    def now(self) -> datetime:
        """เวลาปัจจุบันตาม timezone ของคอนฟิกนี้ (มี tzinfo กำกับเสมอ)"""
        return now(self.tz)

    @property
    def use_edge_tts(self) -> bool:
        return self.tts_backend == "edge"

    @property
    def tts_label(self) -> str:
        """ชื่อเสียงพูดที่กำลังใช้ สำหรับโชว์บนหัวเทอร์มินัลและ chip บนหน้าเว็บ"""
        if self.use_edge_tts:
            from .tts_edge import voice_label   # import ตรงนี้กัน import วนกลับมาที่ config

            return f"Edge · {voice_label(self.edge_voice)}"
        if self.clone_enabled:
            from pathlib import Path

            return f"{self.tts_model} · {Path(self.tts_ref_audio).stem}"
        return self.tts_model

    @property
    def clone_enabled(self) -> bool:
        """โคลนเสียงได้เฉพาะแบ็กเอนด์ api และต่อเมื่อระบุไฟล์อ้างอิงไว้"""
        return not self.use_edge_tts and bool(self.tts_ref_audio)

    @property
    def voice_gender(self) -> str:
        """เพศของเสียงที่กำลังใช้พูด — ตัดสินคำลงท้าย (ครับ/ค่ะ) ในคำตอบของโมเดล

        แบ็กเอนด์ api ไม่มีตัวเลือกเสียงในตัว เพศจึงมาจากไฟล์เสียงอ้างอิงที่โคลนอยู่
        (TTS_REF_GENDER) และตกกลับไปเป็นชายเมื่อไม่ได้โคลน เพราะเสียงสุ่มของ
        OmniVoice ไม่มีเพศแน่นอนอยู่แล้ว
        """
        if self.use_edge_tts:
            from .tts_edge import voice_gender as _voice_gender

            return _voice_gender(self.edge_voice)
        return self.tts_ref_gender if self.clone_enabled else "male"

    @property
    def tts_voice_value(self) -> str:
        """ค่าเดียวที่บอกทั้งแบ็กเอนด์และเสียง — ใช้คุยกับหน้าเว็บ"""
        return f"edge:{self.edge_voice}" if self.use_edge_tts else "api"

    def set_tts_voice(self, value: str) -> bool:
        """รับค่าจากหน้าเว็บ ("api" / "edge:<voice>") คืน True เมื่อเปลี่ยนจริง

        ค่ามาจากเบราว์เซอร์จึงตรวจรูปแบบก่อน — ชื่อเสียงถูกส่งต่อเข้า SSML
        ของ edge-tts ปล่อยข้อความอิสระผ่านไปไม่ได้
        """
        value = (value or "").strip()
        if value == "api":
            if not self.tts_model:
                return False
            changed = self.use_edge_tts
            self.tts_backend = "api"
            return changed
        if value.startswith("edge:"):
            voice = value[5:].strip()
            if not VOICE_NAME.match(voice):
                return False
            changed = not self.use_edge_tts or voice != self.edge_voice
            self.tts_backend = "edge"
            self.edge_voice = voice
            return changed
        return False

    def system_message(self) -> dict:
        """ประกอบ system prompt: กติกาเครื่องมือ → วันเวลาปัจจุบัน → บุคลิกที่ผู้ใช้ตั้ง

        ลำดับสำคัญมาก — ถ้าเอากติกาเครื่องมือไปต่อท้าย โมเดลจะมองข้ามแล้วเดาคำตอบเอง
        (ทดสอบแล้ว: ต่อท้าย = ไม่เรียกค้นเลย · ขึ้นต้น = เรียกถูกจังหวะทุกครั้ง)
        """
        # ต้องระบุ timezone: บน container ที่ TZ=UTC `datetime.now()` เปล่าให้เวลา
        # เพี้ยนไป 7 ชั่วโมง แล้วโมเดลตอบเรื่อง "ตอนนี้" ผิดวันไปเลย (P3-20)
        active_zone = zone(self.tz)
        active_name = active_zone.key
        stamp_at = now(active_zone)
        where = ("ตามเวลาประเทศไทย" if active_name == DEFAULT_TZ
                 else f"ตามเขตเวลา {active_name}")
        stamp = (f"ตอนนี้คือ{THAI_DAYS[stamp_at.weekday()]}ที่ {stamp_at.day} "
                 f"{THAI_MONTHS[stamp_at.month - 1]} พ.ศ. {stamp_at.year + 543} "
                 f"(ค.ศ. {stamp_at.year}) เวลา {stamp_at:%H:%M} น. {where}")
        parts = []
        if self.web_search:
            parts.append(TOOL_INSTRUCTION)
        parts.append(_persona_instruction(self.voice_gender))
        parts.append(stamp)
        parts.append(self.system_prompt)
        return {"role": "system", "content": "\n\n".join(parts)}


SLOW_SINGLE_REQUEST_CHARS = 300     # เกินนี้พร้อม TTS_SINGLE_REQUEST=1 = รอนานผิดปกติ
# วัดจากบันทึกจริงของ OmniVoice: 233 ตัวอักษร = 4.8 วิ · 389 = 14.2 วิ · 608 = 16.1 วิ
SECONDS_PER_CHAR = 0.027


def check_combinations(cfg: Config) -> list[str]:
    """เตือนคู่ค่าที่แต่ละตัวถูกต้องแต่รวมกันแล้วผู้ใช้เจอปัญหา (P3-19 ข้อ 2)"""
    out = cfg.warnings

    if cfg.tts_single_request and cfg.tts_max_chars > SLOW_SINGLE_REQUEST_CHARS:
        wait = int(cfg.tts_max_chars * SECONDS_PER_CHAR)
        warn(f"TTS_SINGLE_REQUEST=1 ร่วมกับ TTS_MAX_CHARS={cfg.tts_max_chars} "
             f"— ระบบจะรอคำตอบจบแล้วสังเคราะห์เสียงทีเดียว ผู้ใช้อาจเงียบรอ "
             f"ราว {wait} วินาทีก่อนได้ยินเสียงแรก (จากบันทึกจริง 608 ตัวอักษร "
             f"ใช้เวลา 16 วินาที) — ตั้ง TTS_SINGLE_REQUEST=0 หรือลด TTS_MAX_CHARS "
             f"ให้ไม่เกิน {SLOW_SINGLE_REQUEST_CHARS}", out)

    if cfg.tts_first_chars > cfg.tts_chunk_chars:
        warn(f"TTS_FIRST_CHARS={cfg.tts_first_chars} มากกว่า "
             f"TTS_CHUNK_CHARS={cfg.tts_chunk_chars} — ก้อนแรกที่ควรสั้นที่สุด "
             f"กลายเป็นก้อนที่ยาวสุด ทำให้เริ่มพูดช้ากว่าที่ควร", out)

    if cfg.tts_chunk_chars > cfg.tts_max_chars:
        warn(f"TTS_CHUNK_CHARS={cfg.tts_chunk_chars} มากกว่า "
             f"TTS_MAX_CHARS={cfg.tts_max_chars} — เพดานต่อคำขอจะเป็นตัวตัดสินจริง "
             f"ค่า TTS_CHUNK_CHARS จึงไม่มีผล", out)

    if cfg.vad_min_utterance_ms >= cfg.vad_max_utterance_ms:
        warn(f"VAD_MIN_UTTERANCE_MS={cfg.vad_min_utterance_ms} ไม่ต่ำกว่า "
             f"VAD_MAX_UTTERANCE_MS={cfg.vad_max_utterance_ms} — "
             f"จะไม่มีคำพูดใดผ่านเกณฑ์เลย ระบบจะเงียบเหมือนไมค์เสีย", out)

    if cfg.vad_abs_threshold > 0.1:
        warn(f"VAD_ABS_THRESHOLD={cfg.vad_abs_threshold} สูงมาก "
             f"(ค่าใช้งานปกติ 0.008-0.03) — เสียงพูดปกติอาจไม่ถึงเกณฑ์เลย "
             f"ตรวจด้วย `python3 voice_chat.py --mic-check` ก่อน", out)

    # ~2 ตัวอักษรไทยต่อ 1 token: ถ้าเพดานฝั่งโมเดลต่ำกว่าเพดานฝั่งเรามาก
    # คำตอบจะถูกตัดโดยโมเดลก่อนถึง guardrail ของเรา แล้วประโยคจะค้างกลางคำ
    if cfg.reply_max_chars > cfg.max_tokens * 2:
        warn(f"REPLY_MAX_CHARS={cfg.reply_max_chars} สูงกว่าที่ "
             f"CHAT_MAX_TOKENS={cfg.max_tokens} จะผลิตได้ (~{cfg.max_tokens * 2} "
             f"ตัวอักษรไทย) — คำตอบจะถูกตัดโดยฝั่งโมเดลและอาจค้างกลางประโยค "
             f"ให้เพิ่ม CHAT_MAX_TOKENS หรือลด REPLY_MAX_CHARS", out)

    return out


def _ref_audio() -> str:
    """ว่าง/0/off = ปิดการโคลน กลับไปใช้เสียงสุ่มของ OmniVoice"""
    raw = os.environ.get("TTS_REF_AUDIO")
    if raw is None:
        return "sound/thai_default.wav"
    raw = raw.strip()
    return "" if raw.lower() in ("", "0", "off", "none") else raw


def _ref_gender() -> str:
    g = (os.environ.get("TTS_REF_GENDER") or "female").strip().lower()
    if g not in ("male", "female"):
        raise SystemExit(f"TTS_REF_GENDER ต้องเป็น male หรือ female เท่านั้น (ได้ '{g}')")
    return g


def load_config() -> Config:
    load_dotenv(ROOT / ".env")

    base = (os.environ.get("API_BASE_URL") or "").rstrip("/")
    key = os.environ.get("API_KEY") or ""
    missing = [k for k, v in (("API_BASE_URL", base), ("API_KEY", key)) if not v]
    if missing:
        raise SystemExit(f"ไม่พบค่าใน .env: {', '.join(missing)}")

    backend = (os.environ.get("TTS_BACKEND") or "api").strip().lower()
    if backend not in ("api", "edge"):
        raise SystemExit(f"TTS_BACKEND ต้องเป็น api หรือ edge เท่านั้น (ได้ '{backend}')")

    w: list[str] = []
    raw_tz = (os.environ.get("APP_TZ") or "").strip() or DEFAULT_TZ
    resolved_tz = zone(raw_tz).key
    cfg = Config(
        warnings=w,
        base_url=base,
        api_key=key,
        chat_model=os.environ.get("CHAT_MODEL") or "claude-sonnet-5",
        asr_model=os.environ.get("ASR_MODEL") or "",
        tts_model=os.environ.get("TTS_MODEL") or "",
        temperature=_f("CHAT_TEMPERATURE", 0.6, w),
        max_tokens=_i("CHAT_MAX_TOKENS", 350, w),
        http_retry_max=_i("HTTP_RETRY_MAX", 2, w),
        http_retry_base_ms=_i("HTTP_RETRY_BASE_MS", 300, w),
        mic_gain=_f("MIC_GAIN", 0.0, w),
        mic_target_noise=_f("MIC_TARGET_NOISE", 0.0035, w),
        vad_abs_threshold=_f("VAD_ABS_THRESHOLD", 0.012, w),
        vad_noise_mult=_f("VAD_NOISE_MULT", 3.2, w),
        vad_confirm_ms=_i("VAD_CONFIRM_MS", 120, w),
        vad_confirm_ms_playback=_i("VAD_CONFIRM_MS_PLAYBACK", 340, w),
        vad_end_silence_ms=_i("VAD_END_SILENCE_MS", 650, w),
        vad_min_utterance_ms=_i("VAD_MIN_UTTERANCE_MS", 350, w),
        vad_max_utterance_ms=_i("VAD_MAX_UTTERANCE_MS", 25000, w),
        echo_guard=_b("ECHO_GUARD", True, w),
        echo_margin=_f("ECHO_MARGIN", 3.5, w),
        barge_in_min_chars=_i("BARGE_IN_MIN_CHARS", 2, w),
        lang_hint=os.environ.get("LANG_HINT") or "th",
        system_prompt=os.environ.get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        tts_single_request=_b("TTS_SINGLE_REQUEST", False, w),
        tts_max_chars=_i("TTS_MAX_CHARS", 260, w),
        tts_first_chars=_i("TTS_FIRST_CHARS", 24, w),
        tts_chunk_chars=_i("TTS_CHUNK_CHARS", 60, w),
        tts_chunk_growth=_f("TTS_CHUNK_GROWTH", 1.8, w),
        tts_search_filler=_b("TTS_SEARCH_FILLER", True, w),
        tts_read_numbers=_b("TTS_READ_NUMBERS", True, w),
        tts_ref_audio=_ref_audio(),
        tts_ref_text=(os.environ.get("TTS_REF_TEXT") or "").strip(),
        tts_ref_max_sec=_f("TTS_REF_MAX_SEC", 0.0, w),
        tts_ref_normalize=_b("TTS_REF_NORMALIZE", True, w),
        tts_ref_gender=_ref_gender(),
        tts_backend=backend,
        edge_voice=os.environ.get("EDGE_TTS_VOICE") or "th-TH-PremwadeeNeural",
        edge_rate=os.environ.get("EDGE_TTS_RATE") or "+0%",
        edge_volume=os.environ.get("EDGE_TTS_VOLUME") or "+0%",
        edge_pitch=os.environ.get("EDGE_TTS_PITCH") or "+0Hz",
        reply_max_sentences=_i("REPLY_MAX_SENTENCES", 4, w),
        reply_max_chars=_i("REPLY_MAX_CHARS", 320, w),
        keep_findings=_i("KEEP_FINDINGS", 4, w),
        tz=resolved_tz,
        log_dir=log_dir(),
        log_transcript=_b("LOG_TRANSCRIPT", True, w),
        log_retention_days=_i("LOG_RETENTION_DAYS", 0, w),
        web_search=_b("WEB_SEARCH", True, w),
        search_results=_i("SEARCH_RESULTS", 5, w),
        search_timeout=_f("SEARCH_TIMEOUT", 12.0, w),
        fetch_max_chars=_i("FETCH_MAX_CHARS", 3000, w),
        fetch_max_bytes=_i("FETCH_MAX_BYTES", 512_000, w),
        tool_rounds=_i("TOOL_ROUNDS", 2, w),
    )
    check_combinations(cfg)
    return cfg
