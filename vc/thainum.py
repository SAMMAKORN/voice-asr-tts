"""อ่านตัวเลขเป็นตัวหนังสือไทยก่อนส่งเข้า TTS

k2-fsa/OmniVoice อ่านตัวเลขอารบิกไม่ออก — "25" กลายเป็นเสียงเงียบหรือเสียงมั่ว
จึงต้องแปลงเป็นคำไทยก่อนสังเคราะห์เสียง (`ยี่สิบห้า`)

แปลงเฉพาะสายที่ไปเข้า TTS เท่านั้น (`clean_for_tts`) — ข้อความบนหน้าจอ, ประวัติ
สนทนาที่ส่งกลับให้โมเดล และ transcript ยังเป็นตัวเลขปกติทุกจุด

กฎการอ่านที่ใช้ (ตามหลักการอ่านตัวเลขภาษาไทย)
  * หลักหน่วยเป็น 1 เมื่อมีหลักอื่นนำหน้า อ่าน "เอ็ด"  — 21 = ยี่สิบเอ็ด
  * หลักสิบเป็น 1 อ่าน "สิบ" เฉย ๆ, เป็น 2 อ่าน "ยี่สิบ"
  * เกินล้านอ่านซ้ำคำ "ล้าน" — 10¹² = หนึ่งล้านล้าน
  * ทศนิยมอ่าน "จุด" แล้วไล่ตัวเลขทีละตัว — 3.14 = สามจุดหนึ่งสี่
  * เลขที่ขึ้นต้นด้วยศูนย์ (เบอร์โทร) อ่านทีละตัว — 081 = ศูนย์แปดหนึ่ง

สิ่งที่ตั้งใจไม่แตะ: ขีดกลางระหว่างเลข ("10-20", "2024-01-15") ไม่มีกฎที่แยก
ช่วงตัวเลขออกจากวันที่ได้แน่นอน จะอ่านว่า "ถึง" ก็ผิดครึ่งหนึ่ง จึงปล่อยไว้ตามเดิม
"""
from __future__ import annotations

import re

DIGITS = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
# ชื่อหลักภายในหนึ่งกลุ่มล้าน (index = ตำแหน่งนับจากหลักหน่วย)
POSITIONS = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน")
GROUP = 1_000_000        # ขนาดหนึ่งกลุ่มก่อนขึ้นคำว่า "ล้าน"
MILLION = "ล้าน"
POINT = "จุด"
MINUS = "ลบ"
PERCENT = "เปอร์เซ็นต์"
# ยาวเกินนี้ไม่มีใครอ่านเป็นจำนวนอีกแล้ว (เลขบัตรประชาชน 13 หลัก, เลขที่บัญชี)
# → อ่านทีละตัว ยกเว้นเลขที่เขียนคั่นหลักพันมา ซึ่งบอกอยู่แล้วว่าเป็น "จำนวน"
DIGIT_BY_DIGIT_LEN = 12

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# ตัวเลขหนึ่งก้อน: เครื่องหมายลบ (ต้องไม่ติดคำ/เลขข้างหน้า ไม่งั้น "10-20" หรือ
# "081-234" จะกลายเป็นค่าลบ), หลักพันคั่นคอมมา, ทศนิยม, และ % ที่ติดท้าย
NUMBER = re.compile(
    r"(?P<sign>(?<![\w])-)?"
    r"(?P<int>[0-9]{1,3}(?:,[0-9]{3})+(?![0-9,])|[0-9]+)"
    r"(?:\.(?P<frac>[0-9]+))?"
    r"(?P<pct>\s*%)?"
)
# เบอร์โทรที่คั่นด้วยขีด/วรรค (081-234-5678) — ต้องอ่านทีละตัวทั้งเบอร์ ไม่ใช่
# "สองร้อยสามสิบสี่" กลางเบอร์ · เบอร์ที่เขียนติดกันเข้าเกณฑ์ "ขึ้นต้นด้วยศูนย์" อยู่แล้ว
PHONE = re.compile(r"(?<![\w-])0\d{1,3}(?:[- ]\d{3,8})+(?![\w-])")
PHONE_MIN_DIGITS = 9        # เบอร์บ้าน 9 หลัก · มือถือ 10 หลัก · สั้นกว่านี้ไม่ใช่เบอร์
# เวลาแบบ 14:30 — อ่านเป็นนาฬิกา/นาที ไม่ใช่ "สิบสี่ สามสิบ"
# ต้องมีนาทีสองหลักเสมอ จึงไม่ไปชนอัตราส่วนอย่าง "2:1"
TIME = re.compile(r"(?<![0-9.:])([01]?[0-9]|2[0-3]):([0-5][0-9])(?![0-9:])")


def _read_group(value: int, *, ed: bool) -> str:
    """อ่านเลขหนึ่งกลุ่ม (1..999,999) — `ed` = หลักหน่วยที่เป็น 1 ต้องอ่าน "เอ็ด" """
    digits = str(value)
    out: list[str] = []
    for offset, char in enumerate(digits):
        digit = int(char)
        if not digit:
            continue
        position = len(digits) - 1 - offset
        if position == 0:
            out.append("เอ็ด" if digit == 1 and ed else DIGITS[digit])
        elif position == 1:
            out.append("สิบ" if digit == 1
                       else "ยี่สิบ" if digit == 2
                       else DIGITS[digit] + "สิบ")
        else:
            out.append(DIGITS[digit] + POSITIONS[position])
    return "".join(out)


def read_int(value: int) -> str:
    """อ่านจำนวนเต็มเป็นคำไทย (รับค่าลบได้)"""
    if value < 0:
        return MINUS + read_int(-value)
    if value == 0:
        return DIGITS[0]
    groups: list[int] = []
    while value:
        groups.append(value % GROUP)
        value //= GROUP
    out: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group:
            # "เอ็ด" ใช้เมื่อมีหลักอื่นนำหน้า — ในกลุ่มเดียวกัน (>9) หรือกลุ่มที่ใหญ่กว่า
            out.append(_read_group(group, ed=group > 9 or index < len(groups) - 1))
        if index:
            # ต้องเติมทุกขอบกลุ่มแม้กลุ่มนั้นเป็นศูนย์ทั้งกลุ่ม ไม่งั้น 10¹²
            # ("หนึ่งล้านล้าน") จะเหลือแค่ "หนึ่งล้าน"
            out.append(MILLION)
    return "".join(out)


def read_digits(digits: str) -> str:
    """อ่านไล่ทีละตัว — ใช้กับเบอร์โทร/เลขอ้างอิงที่ไม่ใช่ "จำนวน" """
    return "".join(DIGITS[int(ch)] for ch in digits if ch.isdigit())


def read_number(int_part: str, frac: str = "", *, negative: bool = False) -> str:
    """อ่านตัวเลขหนึ่งก้อนที่แยกส่วนมาแล้ว (int_part อาจมีคอมมาคั่นหลักพัน)"""
    grouped = "," in int_part
    body = int_part.replace(",", "")
    if not grouped and ((len(body) > 1 and body[0] == "0")
                        or len(body) > DIGIT_BY_DIGIT_LEN):
        # ขึ้นต้นด้วยศูนย์ = เบอร์โทร/รหัส ไม่ใช่จำนวน · ยาวเกินไป = เลขอ้างอิง
        head = read_digits(body)
    else:
        head = read_int(int(body))
    if frac:
        head += POINT + read_digits(frac)
    return (MINUS if negative else "") + head


def _phone_repl(m: re.Match[str]) -> str:
    raw = m.group(0)
    if sum(ch.isdigit() for ch in raw) < PHONE_MIN_DIGITS:
        return raw          # สั้นเกินกว่าจะเป็นเบอร์ — ปล่อยให้ NUMBER จัดการต่อ
    # เว้นวรรคตามขีดเดิม เพื่อให้ TTS เว้นจังหวะเป็นกลุ่ม ๆ เหมือนคนอ่านเบอร์
    return " ".join(read_digits(part) for part in re.split(r"[- ]", raw))


def _time_repl(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    out = read_int(hour) + "นาฬิกา"
    if minute:
        out += read_int(minute) + "นาที"
    return out


def _number_repl(m: re.Match[str]) -> str:
    out = read_number(m.group("int"), m.group("frac") or "",
                      negative=bool(m.group("sign")))
    if m.group("pct"):
        out += PERCENT
    return out


def speak_numbers(text: str) -> str:
    """แทนตัวเลขทุกก้อนในข้อความด้วยคำอ่านภาษาไทย"""
    if not text:
        return text
    text = text.translate(THAI_DIGITS)
    if not any(ch.isdigit() for ch in text):
        return text
    text = PHONE.sub(_phone_repl, text)
    text = TIME.sub(_time_repl, text)
    return NUMBER.sub(_number_repl, text)
