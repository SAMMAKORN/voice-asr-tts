"""ตัดข้อความที่สตรีมมาจาก LLM เป็นก้อนสั้น ๆ เพื่อส่ง TTS ทันที

ภาษาไทยไม่เว้นวรรคระหว่างคำและมักไม่มีจุด จึงตัดที่:
  * เครื่องหมายวรรคตอน (จุด, ?, !, ขึ้นบรรทัดใหม่, ฯลฯ)
  * ช่องว่าง/วรรค เมื่อความยาวถึงเกณฑ์
  * ตัดตรง ๆ เมื่อยาวเกินเพดาน
ก้อนแรกตั้งเกณฑ์สั้นเป็นพิเศษ เพื่อให้ AI เริ่มพูดเร็วที่สุด
"""
from __future__ import annotations

import re

STRONG = ".!?…。！？\n:;"
SOFT = " \t,)]”\"'ๆ"
# ขอบ "ประโยค" จริง ๆ (แคบกว่า STRONG: ไม่นับ : กับ ; ที่มักอยู่กลางประโยค)
SENTENCE_END = ".!?…。！？\n"

_MD = re.compile(r"(\*\*|__|\*|`{1,3}|^#{1,6}\s+|^\s*[-•*]\s+|^\s*>\s+)", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_for_tts(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _MD.sub("", text)
    text = text.replace("—", " ").replace("–", " ").replace("|", " ")
    text = _SPACES.sub(" ", text)
    return text.strip()


def split_for_tts(text: str, limit: int) -> list[str]:
    """แบ่งคำตอบเป็นก้อน "ใหญ่ที่สุดเท่าที่ยังไม่เกิน limit" ตัดที่รอยต่อประโยค

    ตรงข้ามกับ SentenceChunker ที่ตัดสั้นเข้าไว้เพื่อให้เริ่มพูดเร็ว — ตัวนี้ใช้ตอน
    ต้องการให้ทั้งคำตอบเป็นเสียงคนเดียวกัน (OmniVoice สุ่มเสียงใหม่ทุก request)
    จึงอยากได้ก้อนน้อยที่สุด แต่ยังต้องมีเพดานเพราะเวลาสังเคราะห์โตเร็วกว่าความยาว
    ข้อความมาก (วัดจริง: 233 ตัวอักษร = 4.8 วินาที แต่ 389 ตัวอักษร = 14.2 วินาที)
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = max((window.rfind(c) for c in STRONG), default=-1)
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit - 1
        piece, text = text[:cut + 1].strip(), text[cut + 1:].strip()
        if piece:
            out.append(piece)
    if text:
        out.append(text)
    return out


class ReplyLimiter:
    """บังคับเพดานความยาวคำตอบระหว่างที่ยังสตรีมอยู่ (P2-15)

    system prompt สั่ง "ไม่เกิน 4 ประโยค" ไว้แล้ว แต่วัดจากบันทึกจริง 126 ข้อความ
    ได้ p90 = 366 ตัวอักษร และยาวสุด 815 → prompt เดียวเอาไม่อยู่ ต้องบังคับในโค้ด

    วิธีตัดที่ไม่ทำให้ข้อความค้างกลางคำ: ปล่อยข้อความออกได้ทันทีตราบใดที่ยังห่าง
    จากเพดานเกิน `hold` ตัวอักษร พอเข้าเขตใกล้เพดานจะกลั้นไว้ในบัฟเฟอร์แล้วปล่อย
    ทีละ "ประโยคที่จบแล้ว" เท่าที่ยังไม่เกินเพดาน ถ้าไม่มีขอบประโยคเลยจนชนเพดาน
    จึงตัดที่ช่องว่างท้ายสุดที่ยังอยู่ในเพดาน

    ปิดการทำงาน: ตั้ง `max_chars` สูงมาก (เช่น 9999) เพื่อปิดเพดานตัวอักษร และ
    `max_sentences = 0` เพื่อปิดเพดานจำนวนประโยค
    """

    def __init__(self, max_sentences: int = 4, max_chars: int = 320,
                 hold: int = 80):
        self.max_sentences = max(0, int(max_sentences))
        self.max_chars = max(0, int(max_chars))
        self.hold = min(hold, self.max_chars // 2) if self.max_chars else 0
        self.sentences = 0        # จำนวนประโยคที่ปล่อยออกไปครบแล้ว
        self.released = 0         # จำนวนตัวอักษรที่ปล่อยออกไปแล้ว
        self.capped = False       # ถึงเพดานแล้ว (ผู้เรียกควรปิดสตรีม)
        self._buf = ""
        self._at_edge = True      # ตอนนี้อยู่ที่รอยต่อประโยคพอดีหรือยัง

    @property
    def enabled(self) -> bool:
        return bool(self.max_chars or self.max_sentences)

    @property
    def _budget(self) -> int:
        """ตัวอักษรที่ยังปล่อยได้อีก (ไม่จำกัดเพดาน = ค่ามหาศาล)"""
        if not self.max_chars:
            return 1 << 30
        return max(0, self.max_chars - self.released)

    def feed(self, delta: str) -> tuple[str, bool]:
        """รับข้อความที่สตรีมมา — คืน (ส่วนที่ปล่อยออกได้, ถึงเพดานแล้วหรือยัง)"""
        if self.capped:
            return "", True
        if not self.enabled:
            self.released += len(delta)
            return delta, False
        self._buf += delta
        out: list[str] = []
        while self._buf and not self.capped:
            piece = self._take()
            if piece is None:
                break
            out.append(piece)
        return "".join(out), self.capped

    def flush(self) -> str:
        """สตรีมจบเอง — ปล่อยส่วนที่กลั้นไว้ออกให้หมด"""
        if self.capped:
            self._buf = ""
            return ""
        piece, self._buf = self._buf, ""
        if piece.strip():
            self.sentences += 1
        self.released += len(piece)
        return piece

    # ------------------------------------------------------------------ ภายใน
    def _take(self) -> str | None:
        """ปล่อยชิ้นถัดไปที่ปล่อยได้อย่างปลอดภัย — None = ยังต้องรอข้อความเพิ่ม"""
        buf = self._buf
        budget = self._budget
        if budget <= 0:
            self.capped = True
            return None

        end = self._sentence_end(buf)
        if end is not None:
            if end <= budget:
                return self._release(end, sentence=True)
            # ประโยคถัดไปยาวเกินเพดานที่เหลือ
            if self._at_edge:
                # ยังไม่ได้ปล่อยเศษของประโยคนี้ออกไป → จบตรงรอยต่อประโยคเดิมได้เลย
                # (ดีกว่าปล่อยเศษออกไปให้เต็มเพดานแล้วค้างกลางประโยค)
                self.capped = True
                self._buf = ""
                return None
            # ปล่อยเศษไปแล้วบางส่วน — ตัดที่ขอบคำท้ายสุดที่ยังอยู่ในเพดาน
            return self._release(self._safe_cut(buf, budget), sentence=True,
                                 capped=True)

        # ยังไม่เจอขอบประโยคในบัฟเฟอร์
        safe = budget - self.hold
        if len(buf) <= safe:
            return self._release(len(buf))          # ยังห่างเพดาน ปล่อยได้เลย
        if len(buf) < budget:
            if safe > 0:
                return self._release(safe)          # ปล่อยเท่าที่ปลอดภัย รอขอบประโยค
            return None                             # อยู่ในเขตกลั้น รอต่อ
        if self._at_edge:
            # ประโยคก่อนหน้าจบพอดีแล้วและประโยคใหม่ยาวจนไม่มีทางจบในเพดาน
            # → หยุดตรงรอยต่อประโยคเดิม ดีกว่าปล่อยเศษออกไปค้างกลางประโยค
            self.capped = True
            self._buf = ""
            return None
        return self._release(self._safe_cut(buf, budget), sentence=True, capped=True)

    def _release(self, n: int, sentence: bool = False,
                 capped: bool = False) -> str:
        piece, self._buf = self._buf[:n], self._buf[n:]
        self.released += len(piece)
        self._at_edge = sentence
        if sentence and piece.strip():
            self.sentences += 1
            if self.max_sentences and self.sentences >= self.max_sentences:
                capped = True
        if capped:
            self.capped = True
            self._buf = ""
        return piece

    @staticmethod
    def _sentence_end(buf: str) -> int | None:
        """ตำแหน่งหลังขอบประโยคแรกที่มีเนื้อความนำหน้าจริง (None = ยังไม่มี)"""
        for i, ch in enumerate(buf):
            if ch not in SENTENCE_END:
                continue
            # จุดทศนิยม/เลขลำดับ ไม่ใช่จบประโยค ("ประมาณ 25.5 องศา")
            if ch == "." and i and buf[i - 1].isdigit():
                # SSE แบ่ง delta ตรงไหนก็ได้ ถ้าก้อนจบที่ "25." ต้องรออักขระ
                # ถัดไปก่อน ไม่งั้นจะนับจุดทศนิยมเป็นจบประโยคและปิด stream เร็วไป
                if i + 1 >= len(buf):
                    return None
                if buf[i + 1].isdigit():
                    continue
            if not buf[:i].strip():
                continue
            end = i + 1
            while end < len(buf) and buf[end] in SENTENCE_END:
                end += 1        # "…" / "?!" / ".\n" นับเป็นขอบเดียว
            return end
        return None

    @staticmethod
    def _safe_cut(buf: str, budget: int) -> int:
        """จุดตัดท้ายสุดที่ยังไม่เกิน budget และไม่ผ่ากลางคำเท่าที่ทำได้"""
        window = buf[:budget]
        for i in range(len(window) - 1, max(0, budget // 3) - 1, -1):
            if window[i] in SOFT or window[i] in SENTENCE_END:
                return i + 1
        return budget


class SentenceChunker:
    """ตัดข้อความที่สตรีมมาเป็นก้อน ๆ โดยก้อนหลัง ๆ ใหญ่ขึ้นเรื่อย ๆ (growth)

    ทำไมต้องโตขึ้น: OmniVoice สุ่มเสียงผู้พูดใหม่ทุก request ยิ่งแบ่งก้อนถี่
    เสียงยิ่งเปลี่ยนคนบ่อย แต่ก้อนแรกต้องสั้นไม่งั้นกว่าจะเริ่มพูดก็นานเกินไป
    ทางออกคือเริ่มสั้นแล้วขยายขึ้น โดยมีเพดานว่าก้อนถัดไปต้องสังเคราะห์เสร็จ
    ก่อนที่ก้อนก่อนหน้าจะเล่นจบ ไม่งั้นเสียงจะขาดเป็นช่วง ๆ แทน
    (วัดจริง: สังเคราะห์ ≈ 1.2 วินาที + 0.02 ต่อตัวอักษร · เสียงยาว ≈ 0.08 ต่อตัวอักษร)
    """

    def __init__(self, first_target: int = 24, target: int = 72, hard_max: int = 170,
                 growth: float = 1.0, max_target: int | None = None):
        self.first_target = first_target
        self.target = target
        self.hard_max = hard_max
        self.growth = max(1.0, growth)
        self.max_target = max_target or target
        self._buf = ""
        self._emitted = 0

    @property
    def _limit(self) -> int:
        if self._emitted == 0:
            return self.first_target
        grown = self.target * self.growth ** (self._emitted - 1)
        return int(min(self.max_target, self.hard_max, grown))

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def _take(self) -> str | None:
        buf = self._buf
        if not buf:
            return None
        limit = self._limit

        cut = -1
        min_len = min(12, limit)
        # 1) เครื่องหมายวรรคตอนที่เจอก่อน = จุดตัดที่ดีที่สุด
        for i, ch in enumerate(buf):
            if ch in STRONG and i + 1 >= min_len:
                cut = i + 1
                break
        # 2) ไม่มีวรรคตอน (ปกติของภาษาไทย) → ตัดที่วรรคท้ายสุดที่ยังไม่เลยเพดานก้อน
        if cut < 0 and len(buf) >= limit:
            lo = max(min_len, limit // 3)
            hi = min(len(buf), int(limit * 1.6))
            for i in range(hi - 1, lo - 1, -1):
                if buf[i] in SOFT:
                    cut = i + 1
                    break
            if cut < 0:
                # ไม่มีวรรคเลยในช่วงนั้น → ใช้วรรคแรกที่เจอถัดไป
                for i in range(hi, len(buf)):
                    if buf[i] in SOFT:
                        cut = i + 1
                        break
        # 3) ยาวเกินเพดานจริง ๆ → ตัดตรง ๆ
        if cut < 0 and len(buf) >= self.hard_max:
            cut = self.hard_max
        if cut < 0:
            return None

        piece, self._buf = buf[:cut], buf[cut:]
        piece = piece.strip()
        if not piece:
            return self._take()
        self._emitted += 1
        return piece

    def flush(self) -> str | None:
        piece = self._buf.strip()
        self._buf = ""
        if piece:
            self._emitted += 1
            return piece
        return None
