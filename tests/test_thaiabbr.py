"""อ่านตัวย่อไทยเป็นคำเต็มก่อนสังเคราะห์เสียง (vc/thaiabbr.py + จุดตัดใน vc/chunker.py)

ตัวย่อสร้างปัญหาสองชั้น ทั้งสองชั้นมาจากจุดที่อยู่ในตัวย่อ
1. TTS อ่าน "ส.ค." ไม่ออก — ต้องคลี่เป็น "สิงหาคม" ก่อน (เฉพาะสายเสียง)
2. จุดในตัวย่อถูกนับเป็นจบประโยค — `ReplyLimiter` ตัดคำตอบทิ้งก่อนเวลาอันควร
   และ `SentenceChunker` ผ่ากลางวันที่จนเสียงเปลี่ยนคนกลางคำ
"""
from __future__ import annotations

import pytest

from vc.chunker import (ReplyLimiter, SentenceChunker, abbrev_dot, clean_for_tts,
                        ordered_marker, splits_abbrev)
from vc.thaiabbr import MONTHS, speak_abbreviations

pytestmark = pytest.mark.unit


# ────────────────────────────────────────────────── คลี่ตัวย่อเป็นคำเต็ม
@pytest.mark.parametrize("short,full", sorted(MONTHS.items()))
def test_every_month_is_spelled_out(short, full) -> None:
    assert speak_abbreviations(f"วันที่ 20 {short} 2568") == f"วันที่ 20 {full} 2568"


@pytest.mark.parametrize("text,expect", [
    ("ปี พ.ศ. 2568", "ปี พุทธศักราช 2568"),
    ("ปี ค.ศ. 1997", "ปี คริสต์ศักราช 1997"),
    ("ระยะ 5 กม.", "ระยะ 5 กิโลเมตร"),
    ("หนัก 2 กก.", "หนัก 2 กิโลกรัม"),
    ("ที่ดิน 120 ตร.ว.", "ที่ดิน 120 ตารางวา"),
    ("บ้าน 200 ตร.ม.", "บ้าน 200 ตารางเมตร"),
    ("ผมอยู่กรุงเทพฯ", "ผมอยู่กรุงเทพมหานคร"),
    ("ปั๊ม ปตท. และบางจาก", "ปั๊ม ปอตอทอ และบางจาก"),
    ("ผัก ผลไม้ ฯลฯ", "ผัก ผลไม้ และอื่น ๆ"),
    # "ฯ" ที่ไม่ได้อยู่ในตารางคือเครื่องหมายว่าตัดข้อความ ไม่ใช่เสียง — ตัดทิ้ง
    ("โปรดเกล้าฯ แต่งตั้ง", "โปรดเกล้า แต่งตั้ง"),
])
def test_known_abbreviations_are_spelled_out(text, expect) -> None:
    assert speak_abbreviations(text) == expect


@pytest.mark.parametrize("text", [
    "ราคา 25.5 บาท",
    "อากาศดีครับ",
    "ดูที่ example.com",
    "",
])
def test_everything_else_is_left_alone(text) -> None:
    assert speak_abbreviations(text) == text


@pytest.mark.parametrize("text,expect", [
    ("20 ส.ค 2568", "20 สิงหาคม 2568"),        # โมเดลเขียนตกจุดท้ายมาเป็นครั้งคราว
    ("ปี พ.ศ 2568", "ปี พุทธศักราช 2568"),
])
def test_a_missing_final_dot_is_still_recognised(text, expect) -> None:
    assert speak_abbreviations(text) == expect


def test_a_unit_without_its_dot_is_left_alone() -> None:
    """"กก" เป็นคำไทยจริง ตัวย่อที่ไม่มีจุดข้างในจึงต้องมีจุดท้ายเสมอ"""
    assert speak_abbreviations("กก มีสองใบ") == "กก มีสองใบ"
    assert speak_abbreviations("หนัก 2 กก.") == "หนัก 2 กิโลกรัม"


def test_the_longest_abbreviation_wins() -> None:
    """"ตร.ม." ต้องไม่ถูก "ม." กินไปก่อน (ตอนนี้ไม่มี "ม." เดี่ยวในตาราง แต่กติกาต้องถูก)"""
    keys = sorted(("ตร.ม.", "ม."), key=len, reverse=True)
    assert keys[0] == "ตร.ม."


# ───────────────────────────── ทั้งเส้นทางเสียง: ต้องไม่เหลือตัวย่อให้ TTS อ่าน
def test_a_date_reaches_tts_as_words_only() -> None:
    spoken = clean_for_tts("ตอนนี้วันที่ 20 ส.ค. 2568 เวลา 14:30 น. ครับ")
    assert spoken == ("ตอนนี้วันที่ ยี่สิบ สิงหาคม สองพันห้าร้อยหกสิบแปด "
                      "เวลา สิบสี่นาฬิกาสามสิบนาที ครับ")
    assert "." not in spoken, "ยังเหลือจุดของตัวย่อให้ TTS อ่านไม่ออก"


def test_the_model_text_itself_is_not_rewritten() -> None:
    """เหมือน TTS_READ_NUMBERS — คลี่ตัวย่อเฉพาะตอนสังเคราะห์เสียง

    `clean_for_tts` ถูกเรียกที่เดียวคือใน `_tts_loop` ข้อความที่ขึ้นจอ เข้าประวัติ
    สนทนา และลง transcript.md เป็นตัวแปรคนละตัวที่ไม่ผ่านฟังก์ชันนี้
    """
    original = "วันที่ 20 ส.ค. 2568"
    clean_for_tts(original)
    assert original == "วันที่ 20 ส.ค. 2568"


# ──────────────────────────────────────── จุดในตัวย่อ ≠ จุดจบประโยค
@pytest.mark.parametrize("text,dots", [
    ("20 ส.ค. 2568", [4, 6]),
    ("ปี พ.ศ. 2568", [4, 6]),
    ("ระยะ 5 กม.", [9]),
    ("อย่าง ปตท. และ", [9]),
    # คำที่ปิดท้ายประโยคจริง ๆ มีสระ/วรรณยุกต์คั่น จึงไม่เข้าเกณฑ์ตัวย่อ
    ("องศาครับ.", []),
    ("หนึ่งครับ.", []),
    ("แล้ว.", []),
    ("อุณหภูมิ 25.5 องศา", []),
    ("see U.S. news", []),
])
def test_which_dots_belong_to_an_abbreviation(text, dots) -> None:
    assert [i for i, ch in enumerate(text) if ch == "." and abbrev_dot(text, i)] == dots


def test_a_buffer_that_starts_mid_word_is_not_mistaken_for_an_abbreviation() -> None:
    """สตรีมปล่อยข้อความไปแล้วบัฟเฟอร์จึงเริ่มกลางคำได้ — "บ." ที่เหลือจาก "ครับ."
    ต้องไม่กลายเป็นตัวย่อ ต้องส่งท้ายของเดิมเข้าไปให้ดูด้วย
    """
    assert splits_abbrev("บ. ต่อ", 2) is True          # ไม่มีบริบท = เดาว่าเป็นตัวย่อ
    assert splits_abbrev("บ. ต่อ", 2, tail="องศาครั") is False


def test_a_date_does_not_burn_the_sentence_budget() -> None:
    """"20 ส.ค. 2568" เคยถูกนับเป็นสามประโยค แล้วคำตอบถูกตัดทิ้งกลางคัน"""
    limiter = ReplyLimiter(max_sentences=2, max_chars=320)
    text = "ตอนนี้วันที่ 20 ส.ค. 2568 ครับ ราคาทองปรับขึ้นเล็กน้อยนะครับ"
    out = ""
    for i in range(0, len(text), 7):
        piece, capped = limiter.feed(text[i:i + 7])
        out += piece
        if capped:
            break
    out += limiter.flush()
    assert out == text, "คำตอบถูกตัดทิ้งเพราะนับจุดของตัวย่อเป็นจบประโยค"
    assert limiter.capped is False


def test_a_chunk_never_ends_in_the_middle_of_an_abbreviation() -> None:
    chunker = SentenceChunker(first_target=24, target=60, growth=1.8)
    chunks = []
    for ch in "ตอนนี้วันที่ 20 ส.ค. 2568 เวลา 14:30 น. ครับ":
        chunks += chunker.feed(ch)
    last = chunker.flush()
    if last:
        chunks.append(last)
    assert all(not c.endswith("ส.") for c in chunks), f"ผ่ากลางตัวย่อ: {chunks}"
    assert "".join(chunks).replace(" ", "") == (
        "ตอนนี้วันที่20ส.ค.2568เวลา14:30น.ครับ".replace(" ", ""))


# ──────────────────────── สิ่งอื่นที่หลุดไปถึง TTS ทั้งที่อ่านออกเสียงไม่ได้
@pytest.mark.parametrize("text,expect", [
    # `_` เดี่ยวเป็นตัวเน้นของ Markdown เหมือน `__` (เจอจริง: "และ_callisto_")
    ("และ_callisto_ ซึ่งพบ", "และcallisto ซึ่งพบ"),
    # เลขหัวข้อต้องอ่านเป็นคำ แต่จุดต้องหายไป ไม่งั้นได้ "หนึ่ง." ที่มีจุดค้าง
    ("1. เรื่องวัน", "หนึ่ง เรื่องวัน"),
    ("2) ข้อสอง", "สอง ข้อสอง"),
    # จุดกลางประโยคที่ตามหลังตัวเลขยังเป็นทศนิยมเหมือนเดิม
    ("ราคา 25.5 บาท", "ราคา ยี่สิบห้าจุดห้า บาท"),
])
def test_markdown_leftovers_never_reach_tts(text, expect) -> None:
    assert clean_for_tts(text) == expect


def test_a_numbered_list_does_not_burn_the_sentence_budget_twice() -> None:
    """"1." ไม่ใช่ประโยค — ขึ้นบรรทัดใหม่นับให้อยู่แล้ว นับซ้ำจะตัดคำตอบตั้งแต่ข้อสอง"""
    text = ("อ๋อ ขออภัยครับ\n1. เรื่องวัน วันพุธนะครับ"
            "\n2. เรื่องราคา ทองขึ้นครับ\n3. เรื่องอื่น ไม่มีครับ")
    limiter = ReplyLimiter(max_sentences=4, max_chars=320)
    out = ""
    for i in range(0, len(text), 7):
        piece, capped = limiter.feed(text[i:i + 7])
        out += piece
        if capped:
            break
    out += limiter.flush()
    assert out == text, "รายการมีลำดับถูกตัดทิ้งกลางคัน"


@pytest.mark.parametrize("text,is_marker", [
    ("1. ก", True),
    ("  2. ก", True),
    ("ก\n3. ข", True),
    ("ราคา 25. ", False),        # กลางประโยค ไม่ใช่ต้นบรรทัด
    ("ราคา 25.5", False),
    ("ครับ.", False),
])
def test_which_dots_belong_to_a_list_marker(text, is_marker) -> None:
    found = any(ch == "." and ordered_marker(text, i)
                for i, ch in enumerate(text))
    assert found is is_marker
