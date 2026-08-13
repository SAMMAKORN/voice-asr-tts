"""ยืนยันว่าเสียงที่ตัดจังหวะ AI เป็นคนพูดจริง ไม่ใช่เสียงลำโพงย้อนเข้าไมค์ (P3-23)

VAD ตัดสินจาก "ระดับเสียง" อย่างเดียว เสียงของ AI เองที่รั่วเข้าไมค์เป็นช่วงสั้น ๆ
จึงนับเป็นการพูดแทรกได้ แล้ว ASR ก็ "เดา" ออกมาเป็นข้อความมั่ว ๆ ผลคือคำตอบจริง
ถูกทิ้งและประวัติสนทนาถูกยัดขยะ

สิ่งที่พบจริงในบันทึก 37 session (34 ครั้งที่มี ASR ตามหลังการพูดแทรก)
* `啥东西？` `我爱你。` `bản thân cậu.` — สั้นและไม่มีอักขระไทยเลย (3 ครั้ง)
* `ครับ` ซ้ำสองครั้ง — เป็นคำท้ายประโยคของ AI เองที่วนกลับเข้าไมค์ (2 ครั้ง)

เกณฑ์เดิมมีแค่ "สั้นกว่า BARGE_IN_MIN_CHARS (=2)" ซึ่งแทบเป็นจริงไม่ได้ และ
"ไม่มีอักขระไทย" ที่จับได้เฉพาะกรณีแรก ตอนนี้เพิ่มข้อสามคือเทียบกับข้อความที่ AI
เพิ่งพูดออกลำโพงจริง ๆ จึงจับกรณี `ครับ` ได้ด้วย — รวมแล้วครอบคลุม 5 ใน 34 ครั้ง
"""
from __future__ import annotations

import re

THAI_CHARS = re.compile(r"[฀-๿]")
# เกินความยาวนี้ ถ้าไม่มีอักขระไทยเลยก็ยังถือว่าเป็นคนพูด (อาจคุยภาษาอื่นจริง)
FOREIGN_MAX_CHARS = 40
# ยาวเกินนี้ไม่เทียบกับเสียงที่ AI พูด — คำที่ยาวขึ้นคือเนื้อหาที่ผู้ใช้ตั้งใจพูด
# (เช่นผู้ใช้ทวนคำว่า "รายละเอียด" ตามที่ AI เพิ่งถาม ต้องนับเป็นการพูดแทรกจริง)
# 8 ตัวอักษรพอดีกับคำท้ายประโยคที่เจอจริง: ครับ · ค่ะ · นะครับ · ได้ครับ
ECHO_MAX_CHARS = 8
_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)

# เหตุผลที่คืนกลับ — ใช้เป็นค่าใน log ด้วย จึงเป็นสตริงคงที่
TOO_SHORT = "too_short"
WRONG_LANGUAGE = "wrong_language"
ECHO_OF_REPLY = "echo_of_reply"


def normalise(text: str) -> str:
    """ตัดช่องว่างและวรรคตอนออกให้เทียบกันได้ (ASR ใส่วรรคตอนไม่เหมือนกันทุกครั้ง)"""
    return _STRIP.sub("", (text or "").strip()).lower()


def noise_reason(text: str, spoken: str = "", *, lang_hint: str = "th",
                 min_chars: int = 2, barged: bool = False) -> str | None:
    """คืนเหตุผลว่าทำไมข้อความนี้ไม่ใช่คำพูดจริง — คืน None ถ้าเป็นคำพูดจริง

    `spoken` คือข้อความที่ออกลำโพงไปแล้วจริงในเทิร์นก่อนหน้า (ไม่ใช่คำตอบเต็ม)
    ใช้เฉพาะตอน `barged=True` เพราะถ้า AI ไม่ได้พูดอยู่ ก็ไม่มีเสียงให้สะท้อน
    """
    t = (text or "").strip()
    if len(t) < min_chars:
        return TOO_SHORT
    if not barged:
        return None
    if not spoken:
        # AI ยังไม่ได้พูดอะไรออกลำโพงเลยในเทิร์นนี้ (เช่นถูกขัดก่อนเสียงแรกจะออก)
        # จึงไม่มีเสียงให้สะท้อนกลับมา ข้อความแปลก ๆ ที่ ASR เดามาต้องเป็นคำพูด
        # จริงของผู้ใช้ (ต่อให้ถอดออกมาไม่ตรง) ไม่ใช่เสียงสะท้อนของ AI เอง
        return None
    if (len(t) < FOREIGN_MAX_CHARS and lang_hint.startswith("th")
            and not THAI_CHARS.search(t)):
        return WRONG_LANGUAGE
    if len(t) <= ECHO_MAX_CHARS:
        needle, haystack = normalise(t), normalise(spoken)
        if needle and haystack and needle in haystack:
            return ECHO_OF_REPLY
    return None
