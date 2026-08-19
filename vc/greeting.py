"""คำทักทายตอนเริ่มระบบ — ให้ LLM แต่งใหม่ทุกครั้ง และไม่ซ้ำของเดิม

เดิมคำทักทายเป็นข้อความตายตัวสี่แบบสุ่มเลือก ผู้ใช้ที่เปิดระบบวันละหลายรอบจึงจำได้
หมดภายในไม่กี่วัน · ตอนนี้ทุกครั้งที่เริ่มระบบจะขอให้ LLM แต่งขึ้นใหม่ พร้อมบอก
วันเวลาปัจจุบันไปด้วย คำทักทายจึงเปลี่ยนตามช่วงเวลาของวันได้เอง

สามข้อที่ทำให้เรื่องนี้ไม่ใช่แค่ "ยิง prompt แล้วเอาผลมาใช้"

1. **ห้ามล้ม** — ไม่มี API key, เน็ตล่ม, โมเดลตอบว่าง: ระบบต้องยังทักทายได้เหมือนเดิม
   ทุกทางที่ผิดพลาดจึงตกกลับไปที่รายการสำรอง (`greeting_text()`) ซึ่งไม่ต้องต่อเน็ต
   รายการเดียวกันนี้ยังถูก `vc/selftest.py` ใช้เป็นข้อความทดสอบ TTS→ASR ด้วย
   จึงต้องคงรูปประโยคที่ผ่านการทดสอบมาแล้วไว้
2. **"ไม่ซ้ำ" ต้องมีความจำข้ามรอบ** — แต่ละครั้งที่เปิดคือ process ใหม่ ตัวแปรใน
   หน่วยความจำจึงจำอะไรไม่ได้เลย · คำทักทายล่าสุด `RECENT_MAX` อันถูกเก็บเป็นไฟล์
   เดียวใต้ `LOG_DIR` แล้วส่งกลับเข้า prompt เป็นรายการ "ห้ามซ้ำ"
3. **ไฟล์นั้นคือคำพูด** — `LOG_TRANSCRIPT=0` แปลว่าห้ามเขียนคำพูดลงดิสก์เลย
   ไฟล์นี้จึงต้องไม่ถูกเขียนด้วย (ผลคือรอบนั้นเลิกกันซ้ำข้ามรอบ แต่ยังแต่งใหม่ทุกครั้ง)
   และถูกสร้างด้วยสิทธิ์ 0600 เหมือนบันทึกอื่นทั้งหมด

คำตอบของโมเดลถูกล้างก่อนใช้เสมอ (`clean()`) — เอาเครื่องหมายคำพูดครอบ, Markdown,
ขึ้นบรรทัดใหม่ และคำนำแบบ "นี่คือคำทักทาย:" ออก เพราะโมเดลใส่มาให้เป็นครั้งคราว
และมันจะถูกอ่านออกเสียงตรง ๆ
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path

from .api import ApiError
from .config import THAI_DAYS, THAI_MONTHS, Config, persona_instruction
from .logger import FILE_MODE, chmod_quiet, private_dir

log = logging.getLogger("voicechat.greeting")

# หลายแบบกันจำเจ — ใช้เมื่อ LLM ใช้ไม่ได้ และเป็นข้อความทดสอบของ vc/selftest.py
GREETINGS_MALE = (
    "สวัสดีครับ ผมพร้อมคุยแล้ว พูดได้เลยครับ พูดแทรกได้ตลอดเวลา",
    "สวัสดีครับ ผมฟังอยู่ครับ พูดมาได้เลย พูดแทรกได้ตลอดเวลานะครับ",
    "หวัดดีครับ พร้อมคุยแล้วครับ อยากถามอะไรพูดได้เลยครับ",
    "สวัสดีครับ วันนี้มีอะไรให้ช่วยไหมครับ พูดแทรกได้ตลอดเวลาเลย",
)
GREETINGS_FEMALE = (
    "สวัสดีค่ะ ดิฉันพร้อมคุยแล้ว พูดได้เลยค่ะ พูดแทรกได้ตลอดเวลา",
    "สวัสดีค่ะ ดิฉันฟังอยู่ค่ะ พูดมาได้เลย พูดแทรกได้ตลอดเวลานะคะ",
    "หวัดดีค่ะ พร้อมคุยแล้วค่ะ อยากถามอะไรพูดได้เลยค่ะ",
    "สวัสดีค่ะ วันนี้มีอะไรให้ช่วยไหมคะ พูดแทรกได้ตลอดเวลาเลย",
)

RECENT_FILE = "greetings.json"   # อยู่ใต้ LOG_DIR ไม่ใช่ในโฟลเดอร์ session
RECENT_MAX = 12                  # จำย้อนหลังกี่อัน (ยาวพอจะไม่ซ้ำในหนึ่งวันทำงาน)
MAX_CHARS = 150                  # ยาวกว่านี้ไม่ใช่คำทักทายแล้ว และ TTS ก็ช้าขึ้นด้วย
ATTEMPTS = 2                     # ได้ของซ้ำมาให้ลองใหม่อีกครั้งก่อนยอมแพ้
# เทียบซ้ำแบบ "คล้ายกันก็ถือว่าซ้ำ" ไม่ใช่ต้องตรงตัวอักษร เพราะโมเดลชอบคืนประโยคเดิม
# ที่ต่างกันคำเดียว ("คุณพูดแทรกได้" กับ "คุณก็พูดแทรกได้") ซึ่งผู้ใช้ฟังแล้วก็คือซ้ำ
SIMILAR = 0.85

# สิ่งที่โมเดลชอบแถมมา แล้วจะถูกอ่านออกเสียงตรง ๆ ถ้าไม่ตัดทิ้ง
_PREFIX = re.compile(r"^\s*(?:คำทักทาย|ข้อความทักทาย|ตัวอย่าง)\s*[:：]\s*", re.I)
_MD = re.compile(r"[*_`#>\[\]]+")
_SPACES = re.compile(r"\s+")
_QUOTES = "\"'“”‘’「」『』《》"


def greeting_text(gender: str, avoid: object = ()) -> str:
    """คำทักทายสำรองแบบไม่ต้องต่อเน็ต — เลี่ยงอันที่เพิ่งใช้ไปถ้ายังมีตัวเลือกเหลือ"""
    pool = GREETINGS_FEMALE if gender == "female" else GREETINGS_MALE
    fresh = [g for g in pool if g not in set(avoid)]
    return random.choice(fresh or list(pool))


def is_repeat(text: str, recent: object) -> bool:
    """ซ้ำหรือคล้ายของที่เคยใช้ไปแล้วไหม (ตัดวรรคทิ้งก่อนเทียบ)"""
    key = "".join(text.split())
    return any(SequenceMatcher(None, key, "".join(old.split())).ratio() >= SIMILAR
               for old in recent)


def clean(text: str) -> str:
    """ตัดสิ่งที่ไม่ควรถูกอ่านออกเสียงทิ้ง แล้วคืนคำทักทายบรรทัดเดียว"""
    text = _MD.sub("", text.strip())
    text = _PREFIX.sub("", text)
    text = _SPACES.sub(" ", text).strip().strip(_QUOTES).strip()
    if len(text) <= MAX_CHARS:
        return text
    # ตัดที่ช่องว่างสุดท้าย ไม่ใช่กลางคำ — ภาษาไทยไม่มีช่องว่างระหว่างคำ การตัดดิบ ๆ
    # ที่ตัวอักษรที่ 150 จึงได้คำครึ่งคำที่ TTS อ่านออกมาเป็นเสียงมั่ว
    head = text[:MAX_CHARS]
    cut = head.rfind(" ")
    return (head[:cut] if cut > MAX_CHARS // 2 else head).strip()


class GreetingWriter:
    """แต่งคำทักทายหนึ่งอันต่อการเริ่มระบบหนึ่งครั้ง"""

    def __init__(self, cfg: Config, api, *, remember: bool = True) -> None:
        self.cfg = cfg
        self.api = api
        # LOG_TRANSCRIPT=0 = ห้ามคำพูดลงดิสก์ ไฟล์ความจำนี้ก็คือคำพูด
        self.remember = remember and cfg.log_transcript
        self.path: Path = cfg.log_dir / RECENT_FILE

    # ------------------------------------------------------------------ ความจำ
    def recent(self) -> list[str]:
        """คำทักทายที่เคยใช้ล่าสุด (ใหม่สุดอยู่ท้าย) — ไฟล์เสีย/ไม่มี = ถือว่าว่าง"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [s for s in data if isinstance(s, str) and s][-RECENT_MAX:]

    def _save(self, text: str) -> None:
        if not self.remember:
            return
        try:
            private_dir(self.cfg.log_dir)
            keep = [g for g in self.recent() if g != text][-(RECENT_MAX - 1):]
            self.path.write_text(json.dumps(keep + [text], ensure_ascii=False),
                                 encoding="utf-8")
            chmod_quiet(self.path, FILE_MODE)
        except OSError as exc:
            log.warning("จำคำทักทายที่ใช้ไปแล้วไม่ได้ (%s)", exc)

    # -------------------------------------------------------------------- prompt
    def _messages(self, recent: list[str]) -> list[dict]:
        at = self.cfg.now()
        stamp = (f"ตอนนี้คือ{THAI_DAYS[at.weekday()]}ที่ {at.day} "
                 f"{THAI_MONTHS[at.month - 1]} เวลา {at:%H:%M} น.")
        rules = [
            "คุณกำลังจะเริ่มบทสนทนาด้วยเสียงกับผู้ใช้ที่พูดภาษาไทย",
            persona_instruction(self.cfg.voice_gender),
            "ตอบกลับมาเป็นตัวคำทักทายอย่างเดียว ห้ามอธิบาย ห้ามใส่เครื่องหมายคำพูดครอบ "
            "ห้ามใส่ Markdown ห้ามใส่อีโมจิ ห้ามขึ้นบรรทัดใหม่",
        ]
        want = [
            f"แต่งคำทักทายเปิดบทสนทนาความยาวไม่เกิน {MAX_CHARS} ตัวอักษร",
            "ต้องบอกให้ผู้ใช้รู้ด้วยว่าพูดแทรกขัดจังหวะได้ตลอดเวลา",
            "เขียนให้เป็นภาษาพูดที่ฟังลื่นหู เพราะข้อความนี้จะถูกอ่านออกเสียง",
            stamp + " ทักทายให้เข้ากับช่วงเวลานี้ได้ถ้าเหมาะ",
        ]
        if recent:
            want.append("ห้ามซ้ำหรือใกล้เคียงกับคำทักทายที่เคยใช้ไปแล้วเหล่านี้:\n"
                        + "\n".join(f"- {g}" for g in recent))
        return [{"role": "system", "content": "\n".join(rules)},
                {"role": "user", "content": "\n".join(want)}]

    def _ask(self, recent: list[str], cancel: threading.Event) -> str:
        parts: list[str] = []
        for piece in self.api.chat_stream(self._messages(recent), cancel):
            parts.append(piece)
            # กันโมเดลที่ไม่ยอมหยุด — คำทักทายยาวเกินเพดานอยู่แล้วตั้งแต่ตรงนี้
            if sum(len(p) for p in parts) > MAX_CHARS * 3:
                break
        return clean("".join(parts))

    # ---------------------------------------------------------------------- ใช้งาน
    def make(self, cancel: threading.Event | None = None) -> str:
        """คืนคำทักทายของรอบนี้ — ตกกลับไปที่รายการสำรองทุกครั้งที่ LLM ใช้ไม่ได้"""
        recent = self.recent()
        if not self.cfg.greet_from_llm or not self.cfg.api_configured:
            return self._fallback(recent)
        stop = cancel if cancel is not None else threading.Event()
        spare = ""        # ของที่โมเดลแต่งมาแล้วแต่ซ้ำ — ยังดีกว่ารายการสำรอง
        for attempt in range(ATTEMPTS):
            if stop.is_set():
                break
            try:
                text = self._ask(recent, stop)
            except ApiError as exc:
                log.warning("แต่งคำทักทายด้วย LLM ไม่สำเร็จ ใช้ข้อความสำรองแทน (%s)", exc)
                break
            except Exception as exc:      # noqa: BLE001 — ห้ามล้มตั้งแต่เริ่มระบบ
                log.warning("แต่งคำทักทายด้วย LLM ไม่สำเร็จ ใช้ข้อความสำรองแทน (%r)", exc)
                break
            if not text or stop.is_set():
                # ถูกพูดแทรก/ปิดแท็บกลางคัน — ข้อความที่ได้อาจขาดครึ่ง และไม่มีใคร
                # ได้ยินมันอยู่ดี จึงต้องไม่จำว่า "เคยใช้ไปแล้ว"
                continue
            if not is_repeat(text, recent):
                self._save(text)
                return text
            spare = spare or text
            log.info("คำทักทายที่ได้ซ้ำของเดิม ลองใหม่ (ครั้งที่ %d)", attempt + 1)
        if spare:
            # ลองครบแล้วยังได้ของคล้ายเดิม — ใช้ของโมเดลไปเถอะ ถอยไปหารายการสำรอง
            # ที่มีอยู่สี่แบบยิ่งซ้ำหนักกว่า
            self._save(spare)
            return spare
        return self._fallback(recent)

    def _fallback(self, recent: list[str]) -> str:
        text = greeting_text(self.cfg.voice_gender, recent)
        self._save(text)
        return text
