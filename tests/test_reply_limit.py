"""P2-15 — เพดานความยาวคำตอบ บังคับในโค้ด ไม่พึ่ง system prompt

ตัวเลขอ้างอิงจากบันทึกจริง 126 ข้อความ: p90 = 366 ตัวอักษร, ยาวสุด 815
ทั้งที่ prompt สั่ง "ห้ามเกิน 4 ประโยค"
"""
from __future__ import annotations

import pytest

from vc.chunker import SENTENCE_END, ReplyLimiter

pytestmark = pytest.mark.unit

# คำตอบยาวจริงจากบันทึก (ย่อรูปแบบไว้: ประโยคไทยยาว ๆ ต่อกันด้วยเว้นวรรค)
LONG = ("ราคาทองวันนี้ปรับขึ้นจากเมื่อวานเล็กน้อยครับ ทองรูปพรรณขายออกอยู่ที่ประมาณ "
        "42,750 บาทต่อบาททอง ส่วนทองแท่งขายออกอยู่ที่ 42,250 บาท. "
        "ราคานี้อ้างอิงจากสมาคมค้าทองคำเมื่อเช้านี้นะครับ. "
        "ถ้าเทียบกับสัปดาห์ที่แล้วถือว่าขึ้นมาราวสามร้อยบาท. "
        "สาเหตุหลักมาจากค่าเงินบาทที่อ่อนลงและราคาทองในตลาดโลกที่ขยับขึ้น. "
        "อยากให้ผมสรุปแนวโน้มช่วงถัดไปให้ฟังด้วยไหมครับ. "
        "หรือถ้าสนใจราคาย้อนหลังผมหาให้ได้เหมือนกันครับ.")


def run(text: str, limiter: ReplyLimiter, step: int = 7) -> str:
    """ป้อนข้อความเข้า limiter แบบทีละก้อนเหมือนที่ LLM สตรีมมาจริง"""
    out = ""
    for i in range(0, len(text), step):
        piece, capped = limiter.feed(text[i:i + step])
        out += piece
        if capped:
            return out
    return out + limiter.flush()


# ───────────────────────────────────────────────────────────── เพดานตัวอักษร
def test_long_reply_is_cut_at_a_sentence_edge() -> None:
    """AC-15.1 — คำตอบ 800 ตัวอักษรต้องถูกตัด ≤ เพดาน และจบที่ขอบประโยค"""
    limiter = ReplyLimiter(max_sentences=99, max_chars=320)
    out = run(LONG, limiter)
    assert len(out) <= 320, f"ยังเกินเพดาน ({len(out)} ตัวอักษร)"
    assert limiter.capped is True
    assert out.strip()[-1] in SENTENCE_END, f"ตัดค้างกลางประโยค: {out[-40:]!r}"
    assert out == LONG[:len(out)], "ข้อความที่ปล่อยออกไปไม่ตรงกับต้นฉบับ"


def test_short_reply_passes_through_unchanged() -> None:
    limiter = ReplyLimiter(max_sentences=4, max_chars=320)
    text = "อากาศวันนี้ร้อนครับ ประมาณ 34 องศา"
    assert run(text, limiter) == text
    assert limiter.capped is False


def test_decimal_point_is_not_a_sentence_end() -> None:
    """ตัวเลขทศนิยมต้องไม่ถูกนับเป็นจบประโยค (ไม่งั้นตัดกลาง '25.5 องศา')"""
    limiter = ReplyLimiter(max_sentences=1, max_chars=320)
    text = "อุณหภูมิ 25.5 องศาครับ. ประโยคที่สองไม่ควรออก."
    out = run(limiter=limiter, text=text)
    assert out.strip() == "อุณหภูมิ 25.5 องศาครับ."


def test_decimal_split_between_stream_deltas_is_not_a_sentence_end() -> None:
    """SSE อาจจบ delta หลังจุดของ `25.` ก่อนเลข 5 จะมาถึง"""
    limiter = ReplyLimiter(max_sentences=1, max_chars=320)
    first, first_capped = limiter.feed("อุณหภูมิ 25.")
    second, second_capped = limiter.feed("5 องศาครับ. ประโยคที่สองไม่ควรออก.")

    assert first_capped is False
    assert second_capped is True
    assert (first + second).strip() == "อุณหภูมิ 25.5 องศาครับ."


def test_sentence_cap_stops_after_n_sentences() -> None:
    limiter = ReplyLimiter(max_sentences=2, max_chars=9999)
    out = run("หนึ่งครับ. สองครับ. สามครับ. สี่ครับ.", limiter)
    assert out.strip() == "หนึ่งครับ. สองครับ."
    assert limiter.sentences == 2
    assert limiter.capped is True


def test_cut_without_any_sentence_edge_falls_back_to_word_edge() -> None:
    """ภาษาไทยมักไม่มีจุด — ต้องตัดที่วรรค ไม่ผ่ากลางคำ"""
    words = ["ทดสอบข้อความยาว", "ต่อเนื่อง", "ไม่มีเครื่องหมายวรรคตอนเลย"]
    text = " ".join(words * 12)
    limiter = ReplyLimiter(max_sentences=0, max_chars=120)
    out = run(text, limiter)
    assert len(out) <= 120
    assert limiter.capped is True
    assert out[-1] == " " or text[len(out)] == " ", (
        f"ตัดกลางคำ: {out[-25:]!r} | ตัวถัดไป {text[len(out):len(out) + 5]!r}")


def test_guardrail_can_be_disabled() -> None:
    """AC-15.3 — ตั้งเพดานตัวอักษรสูงมาก = ไม่มีการตัดด้วยจำนวนตัวอักษร"""
    long_no_dots = "คำตอบยาวมากแต่มีไม่กี่ประโยค " * 30 + "จบครับ."
    limiter = ReplyLimiter(max_sentences=4, max_chars=9999)
    out = run(long_no_dots, limiter)
    assert out == long_no_dots
    assert limiter.capped is False
    assert len(out) > 800, "ตัวอย่างต้องยาวเกิน 800 ตัวอักษรเพื่อพิสูจน์ว่าไม่ถูกตัด"

    both_off = ReplyLimiter(max_sentences=0, max_chars=0)
    assert run(LONG, both_off) == LONG
    assert both_off.enabled is False


def test_twenty_long_replies_never_exceed_the_cap() -> None:
    """AC-15.4 — ยิงคำตอบยาวซ้ำ 20 เทิร์น สัดส่วนที่เกินเพดานต้องเป็น 0%"""
    over = 0
    for i in range(20):
        limiter = ReplyLimiter(max_sentences=4, max_chars=320)
        out = run(LONG[i:] + LONG, limiter, step=3 + i % 5)
        if len(out) > 320:
            over += 1
    assert over == 0


def test_no_text_is_lost_when_not_capped() -> None:
    """ข้อความต้องไม่หายไปเงียบ ๆ ตอนยังไม่ถึงเพดาน (บัฟเฟอร์ต้องถูก flush)"""
    text = "ประโยคเดียวยาวพอประมาณแต่ไม่ถึงเพดานเลยครับ"
    limiter = ReplyLimiter(max_sentences=4, max_chars=len(text) + 30)
    assert run(text, limiter) == text
