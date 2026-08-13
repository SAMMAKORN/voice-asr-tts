"""โหลดค่าตั้งต้นทั้งหมดจาก .env"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# รูปแบบชื่อเสียงของ Microsoft เช่น th-TH-PremwadeeNeural
VOICE_NAME = re.compile(r"^[A-Za-z]{2,3}-[A-Za-z0-9]{2,8}-[A-Za-z0-9]{2,40}$")

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

    # --- แบ็กเอนด์เสียงพูด: "api" = TTS_MODEL บนเซิร์ฟเวอร์ · "edge" = Microsoft Edge ---
    tts_backend: str = "api"
    edge_voice: str = "th-TH-PremwadeeNeural"
    edge_rate: str = "+0%"            # ความเร็ว เช่น +15% เร็วขึ้น, -10% ช้าลง
    edge_volume: str = "+0%"
    edge_pitch: str = "+0Hz"          # ระดับเสียงสูงต่ำ เช่น +20Hz

    save_audio: bool = False
    history_turns: int = 20      # จำนวนข้อความย้อนหลังที่ส่งให้โมเดล
    keep_findings: int = 4       # จำนวนผลค้นเว็บย้อนหลังที่คงไว้ในความจำ
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

    @property
    def use_edge_tts(self) -> bool:
        return self.tts_backend == "edge"

    @property
    def tts_label(self) -> str:
        """ชื่อเสียงพูดที่กำลังใช้ สำหรับโชว์บนหัวเทอร์มินัลและ chip บนหน้าเว็บ"""
        if self.use_edge_tts:
            from .tts_edge import voice_label   # import ตรงนี้กัน import วนกลับมาที่ config

            return f"Edge · {voice_label(self.edge_voice)}"
        return self.tts_model

    @property
    def voice_gender(self) -> str:
        """เพศของเสียงที่กำลังใช้พูด — ตัดสินคำลงท้าย (ครับ/ค่ะ) ในคำตอบของโมเดล

        แบ็กเอนด์ api มีเสียงเดียวตายตัว (TTS_MODEL) ไม่มีตัวเลือกเพศจึงถือเป็นชายเสมอ
        """
        if self.use_edge_tts:
            from .tts_edge import voice_gender as _voice_gender

            return _voice_gender(self.edge_voice)
        return "male"

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
        now = datetime.now()
        stamp = (f"ตอนนี้คือ{THAI_DAYS[now.weekday()]}ที่ {now.day} "
                 f"{THAI_MONTHS[now.month - 1]} พ.ศ. {now.year + 543} "
                 f"(ค.ศ. {now.year}) เวลา {now:%H:%M} น. ตามเวลาประเทศไทย")
        parts = []
        if self.web_search:
            parts.append(TOOL_INSTRUCTION)
        parts.append(_persona_instruction(self.voice_gender))
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

    backend = (os.environ.get("TTS_BACKEND") or "api").strip().lower()
    if backend not in ("api", "edge"):
        raise SystemExit(f"TTS_BACKEND ต้องเป็น api หรือ edge เท่านั้น (ได้ '{backend}')")

    return Config(
        base_url=base,
        api_key=key,
        chat_model=os.environ.get("CHAT_MODEL") or "claude-sonnet-5",
        asr_model=os.environ.get("ASR_MODEL") or "",
        tts_model=os.environ.get("TTS_MODEL") or "",
        temperature=_f("CHAT_TEMPERATURE", 0.6),
        max_tokens=_i("CHAT_MAX_TOKENS", 350),
        mic_gain=_f("MIC_GAIN", 0.0),
        mic_target_noise=_f("MIC_TARGET_NOISE", 0.0035),
        vad_abs_threshold=_f("VAD_ABS_THRESHOLD", 0.012),
        vad_noise_mult=_f("VAD_NOISE_MULT", 3.2),
        vad_confirm_ms=_i("VAD_CONFIRM_MS", 120),
        vad_confirm_ms_playback=_i("VAD_CONFIRM_MS_PLAYBACK", 340),
        vad_end_silence_ms=_i("VAD_END_SILENCE_MS", 650),
        vad_min_utterance_ms=_i("VAD_MIN_UTTERANCE_MS", 350),
        vad_max_utterance_ms=_i("VAD_MAX_UTTERANCE_MS", 25000),
        echo_guard=_b("ECHO_GUARD", True),
        echo_margin=_f("ECHO_MARGIN", 3.5),
        barge_in_min_chars=_i("BARGE_IN_MIN_CHARS", 2),
        lang_hint=os.environ.get("LANG_HINT") or "th",
        system_prompt=os.environ.get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        tts_single_request=_b("TTS_SINGLE_REQUEST", False),
        tts_max_chars=_i("TTS_MAX_CHARS", 260),
        tts_first_chars=_i("TTS_FIRST_CHARS", 24),
        tts_chunk_chars=_i("TTS_CHUNK_CHARS", 60),
        tts_chunk_growth=_f("TTS_CHUNK_GROWTH", 1.8),
        tts_search_filler=_b("TTS_SEARCH_FILLER", True),
        tts_backend=backend,
        edge_voice=os.environ.get("EDGE_TTS_VOICE") or "th-TH-PremwadeeNeural",
        edge_rate=os.environ.get("EDGE_TTS_RATE") or "+0%",
        edge_volume=os.environ.get("EDGE_TTS_VOLUME") or "+0%",
        edge_pitch=os.environ.get("EDGE_TTS_PITCH") or "+0Hz",
        keep_findings=_i("KEEP_FINDINGS", 4),
        web_search=_b("WEB_SEARCH", True),
        search_results=_i("SEARCH_RESULTS", 5),
        search_timeout=_f("SEARCH_TIMEOUT", 12.0),
        fetch_max_chars=_i("FETCH_MAX_CHARS", 3000),
        tool_rounds=_i("TOOL_ROUNDS", 2),
    )
