"""อ่านตัวเลขเป็นตัวหนังสือไทยก่อนส่ง TTS (k2-fsa/OmniVoice อ่านตัวเลขไม่ออก)

สองเรื่องที่ต้องจริงตลอด
1. ข้อความที่ไปเข้า TTS ต้องไม่มีตัวเลขอารบิกเหลืออยู่เลย
2. ข้อความที่ขึ้นหน้าจอ/เข้าประวัติสนทนา ต้องยังเป็นตัวเลขปกติ — การแปลงเกิดที่
   `clean_for_tts` ซึ่งเรียกตอนสังเคราะห์เสียงเท่านั้น ไม่ใช่ตอนเขียนข้อความออก
"""
from __future__ import annotations

import pytest

from vc.chunker import SentenceChunker, clean_for_tts, split_for_tts, splits_number
from vc.thainum import read_digits, read_int, speak_numbers

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────── จำนวนเต็ม
@pytest.mark.parametrize("value,expect", [
    (0, "ศูนย์"),
    (1, "หนึ่ง"),
    (2, "สอง"),
    (9, "เก้า"),
    (10, "สิบ"),               # ไม่ใช่ "หนึ่งสิบ"
    (11, "สิบเอ็ด"),            # หลักหน่วยเป็น 1 → เอ็ด
    (20, "ยี่สิบ"),             # หลักสิบเป็น 2 → ยี่สิบ
    (21, "ยี่สิบเอ็ด"),
    (25, "ยี่สิบห้า"),
    (100, "หนึ่งร้อย"),
    (101, "หนึ่งร้อยเอ็ด"),
    (110, "หนึ่งร้อยสิบ"),
    (999, "เก้าร้อยเก้าสิบเก้า"),
    (1000, "หนึ่งพัน"),
    (1001, "หนึ่งพันเอ็ด"),
    (2567, "สองพันห้าร้อยหกสิบเจ็ด"),
    (10000, "หนึ่งหมื่น"),
    (100000, "หนึ่งแสน"),
    (1000000, "หนึ่งล้าน"),
    (1000001, "หนึ่งล้านเอ็ด"),
    (1234567, "หนึ่งล้านสองแสนสามหมื่นสี่พันห้าร้อยหกสิบเจ็ด"),
    (1000000000, "หนึ่งพันล้าน"),
    # ทุกขอบกลุ่มต้องเติม "ล้าน" แม้กลุ่มกลางเป็นศูนย์ทั้งกลุ่ม
    (10 ** 12, "หนึ่งล้านล้าน"),
    (-5, "ลบห้า"),
])
def test_read_int(value: int, expect: str) -> None:
    assert read_int(value) == expect


def test_read_int_has_no_digits_left() -> None:
    for value in list(range(0, 2000)) + [12345, 987654321, 10 ** 15 + 7]:
        out = read_int(value)
        assert not any(ch.isdigit() for ch in out), (value, out)


def test_read_digits_reads_one_by_one() -> None:
    assert read_digits("081") == "ศูนย์แปดหนึ่ง"


# ──────────────────────────────────────────────────── ตัวเลขในข้อความจริง
@pytest.mark.parametrize("text,expect", [
    ("ราคา 25.5 บาท", "ราคา ยี่สิบห้าจุดห้า บาท"),
    ("วันนี้ 30 องศา", "วันนี้ สามสิบ องศา"),
    ("ประชากร 1,234,567 คน", "ประชากร หนึ่งล้านสองแสนสามหมื่นสี่พันห้าร้อยหกสิบเจ็ด คน"),
    ("เพิ่มขึ้น 12.75%", "เพิ่มขึ้น สิบสองจุดเจ็ดห้าเปอร์เซ็นต์"),
    ("ลด 50 %", "ลด ห้าสิบเปอร์เซ็นต์"),
    ("อุณหภูมิ -3 องศา", "อุณหภูมิ ลบสาม องศา"),
    # "น." ถูกกลืนไปกับเวลา เพราะคำอ่านมี "นาฬิกา" อยู่แล้ว และตัวย่อลอย ๆ อ่านไม่ออก
    ("เวลา 14:30 น.", "เวลา สิบสี่นาฬิกาสามสิบนาที"),
    ("เวลา 09:00 น.", "เวลา เก้านาฬิกา"),
    ("เวลา 14.30 น.", "เวลา สิบสี่นาฬิกาสามสิบนาที"),   # เวลาแบบไทยที่ใช้จุดคั่น
    ("ราคา 14.30 บาท", "ราคา สิบสี่จุดสามศูนย์ บาท"),    # จุดคั่นที่ไม่ใช่เวลา
    # ขึ้นต้นด้วยศูนย์ = รหัส/เบอร์ ไม่ใช่จำนวน
    ("ห้อง 0805", "ห้อง ศูนย์แปดศูนย์ห้า"),
    ("โทร 0812345678", "โทร ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด"),
    ("โทร 081-234-5678", "โทร ศูนย์แปดหนึ่ง สองสามสี่ ห้าหกเจ็ดแปด"),
    ("โทร 02-123-4567", "โทร ศูนย์สอง หนึ่งสองสาม สี่ห้าหกเจ็ด"),
    # เลขไทยต้องอ่านได้เหมือนเลขอารบิก
    ("ปี ๒๕๖๗", "ปี สองพันห้าร้อยหกสิบเจ็ด"),
    ("ไม่มีตัวเลขในประโยคนี้", "ไม่มีตัวเลขในประโยคนี้"),
])
def test_speak_numbers(text: str, expect: str) -> None:
    assert speak_numbers(text) == expect


def test_hyphen_between_numbers_left_alone() -> None:
    """ช่วงตัวเลขกับวันที่แยกจากกันไม่ได้ จึงไม่แปลงขีดเป็น "ถึง" — แต่ต้องอ่านเลขได้"""
    assert speak_numbers("ช่วง 10-20 บาท") == "ช่วง สิบ-ยี่สิบ บาท"
    assert speak_numbers("วันที่ 2024-01-15").count("-") == 2


def test_no_arabic_digits_survive() -> None:
    text = ("ทองรูปพรรณ 42,750 บาท ขึ้น 1.5% จากเมื่อวาน โทร 02-123-4567 "
            "ได้ตั้งแต่ 08:30 ถึง 17:00 ห้อง 0805 เลขที่ 1234567890123")
    assert not any(ch.isdigit() for ch in speak_numbers(text))


def test_empty_and_plain_text_untouched() -> None:
    assert speak_numbers("") == ""
    assert speak_numbers("สวัสดีครับ") == "สวัสดีครับ"


# ────────────────────────────────────────── ต่อกับสายที่ส่งเข้า TTS จริง
def test_clean_for_tts_reads_numbers() -> None:
    assert clean_for_tts("**ราคา 25 บาท**") == "ราคา ยี่สิบห้า บาท"


def test_clean_for_tts_can_be_switched_off() -> None:
    """TTS_READ_NUMBERS=0 ต้องกลับไปส่งตัวเลขดิบให้ TTS เหมือนเดิม"""
    assert clean_for_tts("ราคา 25 บาท", False) == "ราคา 25 บาท"


def test_display_text_keeps_digits() -> None:
    """สิ่งที่ขึ้นหน้าจอคือ argument ที่ส่งเข้ามา — clean_for_tts ต้องไม่แก้ของเดิม"""
    original = "ราคา 25 บาท"
    clean_for_tts(original)
    assert original == "ราคา 25 บาท"


# ──────────────────────────────── ห้ามตัดก้อน TTS ผ่ากลางตัวเลข
@pytest.mark.parametrize("buf,cut,expect", [
    ("25.5", 3, True),        # ตัดหลังจุดทศนิยม
    ("25.", 3, True),         # ตัวคั่นอยู่ท้ายบัฟเฟอร์ — ยังไม่รู้ ต้องรอ
    ("จบแล้ว.", 7, False),     # จุดจบประโยคจริง
    ("1,234", 2, True),       # คอมมาหลักพัน
    ("14:30", 3, True),       # เวลา
    ("12345", 3, True),       # กลางเลขล้วน ๆ
    ("25 บาท", 3, False),     # หลังเลขมีวรรค ตัดได้
])
def test_splits_number(buf: str, cut: int, expect: bool) -> None:
    assert splits_number(buf, cut) is expect


def _stream(text: str, step: int = 5) -> list[str]:
    """ป้อนข้อความเข้า SentenceChunker ทีละก้อนเหมือนที่ LLM สตรีมมาจริง"""
    chunker = SentenceChunker(first_target=24, target=40, hard_max=80)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    rest = chunker.flush()
    if rest:
        out.append(rest)
    return out


def test_chunker_keeps_decimal_together() -> None:
    text = "ราคาทองวันนี้อยู่ที่ 42,750.25 บาทต่อบาททองครับ ขึ้นมา 1.5% จากเมื่อวานนี้"
    chunks = _stream(text)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    spoken = [clean_for_tts(c) for c in chunks]
    # ถ้าก้อนถูกผ่าที่จุดทศนิยม จะได้ "ยี่สิบห้า" ลอยมาเป็นจำนวนแยก
    assert "สี่หมื่นสองพันเจ็ดร้อยห้าสิบจุดสองห้า" in "".join(spoken)
    assert "หนึ่งจุดห้าเปอร์เซ็นต์" in "".join(spoken)


def test_chunker_hard_max_does_not_split_digits() -> None:
    """ไม่มีวรรคตอนเลยจนชนเพดาน ต้องถอยจุดตัดให้ไม่ผ่ากลางเลข"""
    text = "ก" * 25 + "123456789"
    chunker = SentenceChunker(first_target=30, target=30, hard_max=30)
    chunks = chunker.feed(text)
    assert chunks == ["ก" * 25]
    assert not splits_number(text, len(chunks[0]))


def test_split_for_tts_does_not_split_numbers() -> None:
    text = ("อัตราแลกเปลี่ยนวันนี้อยู่ที่ 35.75 บาทต่อดอลลาร์ "
            "ส่วนเงินเยนอยู่ที่ 0.24 บาทต่อเยนครับ")
    pieces = split_for_tts(text, 40)
    spoken = "".join(clean_for_tts(p) for p in pieces)
    assert "สามสิบห้าจุดเจ็ดห้า" in spoken
    assert "ศูนย์จุดสองสี่" in spoken


# ─────────────────── ต่อทั้งเทิร์นจริง: เสียงเป็นคำไทย หน้าจอเป็นตัวเลข
def _tokens(*words: str):
    def chat_stream(messages, cancel, **kwargs):
        for w in words:
            if cancel.is_set():
                return
            yield w
    return chat_stream


def test_turn_speaks_words_but_shows_digits(web_session) -> None:
    """เทิร์นเดียวจริง ๆ: TTS ได้ "ยี่สิบห้า" แต่ delta ที่ส่งขึ้นหน้าเว็บยังเป็น "25" """
    import numpy as np

    session = web_session
    session.speaker.LOST_GRACE = 0.2
    spoken: list[str] = []

    def synthesize(text: str, sr: int):
        spoken.append(text)
        return np.zeros(int(session.cfg.speaker_sr * 0.02), np.int16)

    session.api.transcribe = lambda pcm, sr: "วันนี้อากาศเท่าไหร่ครับ"  # type: ignore[method-assign]
    session.api.chat_stream = _tokens("วันนี้", "กรุงเทพ ", "25 องศาครับ")  # type: ignore[method-assign]
    session.api.synthesize = synthesize                # type: ignore[method-assign]
    session.start_workers()
    session.handle_utterance(np.zeros(int(session.cfg.mic_sr * 0.8), np.int16))

    assert spoken, "ไม่มีข้อความถูกส่งเข้า TTS เลย"
    assert not any(ch.isdigit() for ch in "".join(spoken)), spoken
    assert "ยี่สิบห้า" in "".join(spoken)

    shown = "".join(m.get("text", "") for m in session.out.of("delta"))
    assert "25 องศา" in shown, shown
    answers = [m["content"] for m in session.messages if m["role"] == "assistant"]
    assert "25 องศาครับ" in answers[-1], answers
