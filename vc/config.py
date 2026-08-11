"""โหลดค่าตั้งต้นทั้งหมดจาก .env"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

THAI_DAYS = ("วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี",
             "วันศุกร์", "วันเสาร์", "วันอาทิตย์")
THAI_MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม")


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except ValueError:
        return default


def _i(key: str, default: int) -> int:
    return int(_f(key, default))


def _b(key: str, default: bool) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


DEFAULT_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วย AI ที่คุยด้วยเสียงกับผู้ใช้เป็นภาษาไทยเป็นหลัก\n"
    "กติกาการตอบ (สำคัญมาก เพราะคำตอบจะถูกอ่านออกเสียง):\n"
    "1. ตอบสั้น กระชับ เป็นภาษาพูดที่ฟังลื่นหู ปกติ 1-3 ประโยค ยาวได้เมื่อผู้ใช้ขอรายละเอียด\n"
    "2. ห้ามใช้ Markdown, bullet, ตาราง, emoji หรือสัญลักษณ์พิเศษ เพราะอ่านออกเสียงไม่ได้\n"
    "3. เขียนตัวเลข หน่วย และคำย่อในรูปที่อ่านออกเสียงได้เลย เช่น 'ประมาณ 25 องศา'\n"
    "4. ถ้าคำถามกำกวมหรือข้อความที่ถอดเสียงมาดูเพี้ยน ให้ถามกลับสั้น ๆ เพื่อยืนยัน\n"
    "5. ถ้าถูกผู้ใช้พูดขัดกลางประโยค ให้หยุดเรื่องเดิมทันทีแล้วตอบเรื่องใหม่ที่ผู้ใช้พูด\n"
    "6. ตอบเป็นภาษาไทยเสมอ ยกเว้นผู้ใช้พูดภาษาอื่นหรือขอให้ตอบภาษาอื่น"
)


TOOL_INSTRUCTION = (
    "สำคัญที่สุด: คุณต่ออินเทอร์เน็ตได้ผ่านเครื่องมือ web_search\n"
    "ห้ามตอบเรื่องต่อไปนี้จากความจำของคุณเด็ดขาด ต้องเรียก web_search ก่อนเสมอ — "
    "ราคาทุกชนิด อัตราแลกเปลี่ยน สภาพอากาศ ข่าว ผลกีฬา ตารางเวลา "
    "หรืออะไรก็ตามที่มีคำว่าวันนี้ ตอนนี้ ล่าสุด ปัจจุบัน\n"
    "ความรู้ในตัวคุณเก่าแล้ว การเดาตัวเลขเองถือว่าผิดร้ายแรง "
    "ส่วนคำถามความรู้ทั่วไปที่ไม่เปลี่ยนตามเวลา ตอบเองได้เลยไม่ต้องค้น\n"
    "เมื่อตอบจากผลค้นหา ให้บอกแหล่งที่มาสั้น ๆ แบบภาษาพูด "
    "เช่น 'อ้างอิงจากสมาคมค้าทองคำนะครับ' พร้อมบอกวันที่ของข้อมูลถ้ามี "
    "ห้ามอ่าน URL ออกเสียง"
)


@dataclass
class Config:
    # --- API ---
    base_url: str = ""
    api_key: str = ""
    chat_model: str = "claude-sonnet-5"
    asr_model: str = ""
    tts_model: str = ""
    temperature: float = 0.6
    max_tokens: int = 800

    # --- เสียง ---
    mic_sr: int = 16000          # อัตราสุ่มที่ ASR ต้องการ
    speaker_sr: int = 24000      # อัตราสุ่มที่ OmniVoice คืนมา
    frame_ms: int = 20
    input_device: int | None = None
    output_device: int | None = None

    # --- VAD / barge-in ---
    vad_abs_threshold: float = 0.012
    vad_noise_mult: float = 3.2
    vad_confirm_ms: int = 120
    vad_confirm_ms_playback: int = 260
    vad_end_silence_ms: int = 650
    vad_min_utterance_ms: int = 350
    vad_max_utterance_ms: int = 25000
    echo_guard: bool = True
    echo_margin: float = 3.0

    # --- พฤติกรรม ---
    lang_hint: str = "th"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tts_enabled: bool = True
    save_audio: bool = False
    history_turns: int = 20      # จำนวนข้อความย้อนหลังที่ส่งให้โมเดล
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")

    # --- ค้นข้อมูลจากอินเทอร์เน็ต ---
    web_search: bool = True
    search_results: int = 5
    search_timeout: float = 12.0
    fetch_max_chars: int = 3000
    tool_rounds: int = 2         # จำนวนรอบสูงสุดที่ยอมให้เรียกเครื่องมือ

    @property
    def frame_samples(self) -> int:
        return int(self.mic_sr * self.frame_ms / 1000)

    def system_message(self) -> dict:
        """ประกอบ system prompt: กติกาเครื่องมือ → วันเวลาปัจจุบัน → บุคลิกที่ผู้ใช้ตั้ง

        ลำดับสำคัญมาก — ถ้าเอากติกาเครื่องมือไปต่อท้าย โมเดลจะมองข้ามแล้วเดาคำตอบเอง
        (ทดสอบแล้ว: ต่อท้าย = ไม่เรียกค้นเลย · ขึ้นต้น = เรียกถูกจังหวะทุกครั้ง)
        """
        now = datetime.now()
        stamp = (f"ตอนนี้คือ{THAI_DAYS[now.weekday()]}ที่ {now.day} "
                 f"{THAI_MONTHS[now.month - 1]} พ.ศ. {now.year + 543} "
                 f"(ค.ศ. {now.year}) เวลา {now:%H:%M} น. ตามเวลาประเทศไทย")
        parts = []
        if self.web_search:
            parts.append(TOOL_INSTRUCTION)
        parts.append(stamp)
        parts.append(self.system_prompt)
        return {"role": "system", "content": "\n\n".join(parts)}


def load_config() -> Config:
    load_dotenv(ROOT / ".env")

    base = (os.environ.get("API_BASE_URL") or "").rstrip("/")
    key = os.environ.get("API_KEY") or ""
    missing = [k for k, v in (("API_BASE_URL", base), ("API_KEY", key)) if not v]
    if missing:
        raise SystemExit(f"ไม่พบค่าใน .env: {', '.join(missing)}")

    return Config(
        base_url=base,
        api_key=key,
        chat_model=os.environ.get("CHAT_MODEL") or "claude-sonnet-5",
        asr_model=os.environ.get("ASR_MODEL") or "",
        tts_model=os.environ.get("TTS_MODEL") or "",
        temperature=_f("CHAT_TEMPERATURE", 0.6),
        max_tokens=_i("CHAT_MAX_TOKENS", 800),
        vad_abs_threshold=_f("VAD_ABS_THRESHOLD", 0.012),
        vad_noise_mult=_f("VAD_NOISE_MULT", 3.2),
        vad_confirm_ms=_i("VAD_CONFIRM_MS", 120),
        vad_confirm_ms_playback=_i("VAD_CONFIRM_MS_PLAYBACK", 260),
        vad_end_silence_ms=_i("VAD_END_SILENCE_MS", 650),
        vad_min_utterance_ms=_i("VAD_MIN_UTTERANCE_MS", 350),
        vad_max_utterance_ms=_i("VAD_MAX_UTTERANCE_MS", 25000),
        echo_guard=_b("ECHO_GUARD", True),
        echo_margin=_f("ECHO_MARGIN", 3.0),
        lang_hint=os.environ.get("LANG_HINT") or "th",
        system_prompt=os.environ.get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        web_search=_b("WEB_SEARCH", True),
        search_results=_i("SEARCH_RESULTS", 5),
        search_timeout=_f("SEARCH_TIMEOUT", 12.0),
        fetch_max_chars=_i("FETCH_MAX_CHARS", 3000),
        tool_rounds=_i("TOOL_ROUNDS", 2),
    )
