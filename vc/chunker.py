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

_MD = re.compile(r"(\*\*|__|\*|`{1,3}|^#{1,6}\s+|^\s*[-•*]\s+|^\s*>\s+)", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_for_tts(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _MD.sub("", text)
    text = text.replace("—", " ").replace("–", " ").replace("|", " ")
    text = _SPACES.sub(" ", text)
    return text.strip()


class SentenceChunker:
    def __init__(self, first_target: int = 24, target: int = 72, hard_max: int = 170):
        self.first_target = first_target
        self.target = target
        self.hard_max = hard_max
        self._buf = ""
        self._emitted = 0

    @property
    def _limit(self) -> int:
        return self.first_target if self._emitted == 0 else self.target

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
