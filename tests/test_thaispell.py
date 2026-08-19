"""แก้คำผิดภาษาไทยของผลถอดเสียงก่อนส่งให้โมเดล (vc/thaispell.py, ใช้ PyThaiNLP)

สามเรื่องที่ต้องจริงตลอด เรียงตามความสำคัญ
1. **ประโยคไทยที่สะกดถูกอยู่แล้วต้องไม่ถูกแตะเลย** — ตัวแก้คำผิดที่แก้ของที่ถูกอยู่แล้ว
   ให้ผิด แย่กว่าไม่มีตัวแก้คำผิด เพราะผู้ใช้ไม่มีทางรู้ว่าโมเดลได้ยินคนละคำ
2. แก้ได้เฉพาะรูปวรรณยุกต์/สระบน-ล่าง/การันต์ โครงพยัญชนะห้ามเปลี่ยน
3. ไม่มี PyThaiNLP บนเครื่อง = ทำงานต่อได้ทุกอย่าง แค่ไม่มีการแก้คำ (ห้าม raise)
"""
from __future__ import annotations

import pytest

from vc.thaispell import ThaiSpell, engine, skeleton

pytestmark = pytest.mark.unit

# ทั้งไฟล์ต้องการพจนานุกรมจริง ยกเว้นกลุ่ม "เครื่องที่ไม่มี PyThaiNLP" ท้ายไฟล์
needs_pythainlp = pytest.mark.skipif(
    engine() is None, reason="เครื่องนี้ไม่ได้ติดตั้ง PyThaiNLP")


@pytest.fixture
def spell() -> ThaiSpell:
    return ThaiSpell()


# ─────────────────────────────────── 1. ของที่ถูกอยู่แล้วห้ามถูกแตะ (สำคัญที่สุด)
CORRECT_THAI = [
    "สวัสดีครับ วันนี้อากาศเป็นอย่างไรบ้าง",
    "ช่วยสรุปข่าวเศรษฐกิจล่าสุดให้หน่อยครับ",
    "ขอราคาบ้านมือสองย่านรังสิตหน่อยค่ะ",
    "ตอนนี้กี่โมงแล้วครับ แล้วพรุ่งนี้ฝนจะตกไหม",
    "ดอกเบี้ยสินเชื่อบ้านของธนาคารกสิกรไทยเท่าไหร่",
    "ผัดกะเพราหมูสับไข่ดาวหนึ่งจานครับ",
    "คุณสมชายโทรศัพท์มาหาผมเมื่อเช้านี้",
    "ช่วยค้นหาข้อมูลเรื่องภาษีที่ดินและสิ่งปลูกสร้างให้หน่อย",
    "ขอบคุณมากครับ ไว้คุยกันใหม่พรุ่งนี้",
    "เมื่อวานผมไปประชุมที่สำนักงานใหญ่มา",
]


@needs_pythainlp
@pytest.mark.parametrize("text", CORRECT_THAI)
def test_correct_thai_is_never_touched(spell, text) -> None:
    fixed, changes = spell.fix(text)
    assert changes == [], f"แก้ของที่ถูกอยู่แล้ว: {changes}"
    assert fixed == text


@needs_pythainlp
def test_a_thai_name_is_not_corrected_into_another_word(spell) -> None:
    """ชื่อคนไทยอยู่ในรายการคำที่รู้จัก ไม่งั้น "สมชาย" กลายเป็น "ลมชาย" """
    assert spell.fix("สมชาย")[0] == "สมชาย"
    assert spell.fix("สมหญิง")[0] == "สมหญิง"


# ─────────────────────────────────────────────── 2. คำที่ควรแก้ได้จริง
@needs_pythainlp
@pytest.mark.parametrize("before,after", [
    ("เขาเปนคนดี", "เขาเป็นคนดี"),                       # ไม้ไต่คู้หาย
    ("ขอทราบรายละเอียดเพิมเติม", "ขอทราบรายละเอียดเพิ่มเติม"),   # ไม้เอกหาย
    ("ผมจะไปบ้านเพือน", "ผมจะไปบ้านเพื่อน"),
    ("ฉนัไม่เข้าใจคำถาม", "ฉันไม่เข้าใจคำถาม"),           # สระสลับที่
])
def test_misspellings_are_fixed(spell, before, after) -> None:
    assert spell.fix(before)[0] == after


@needs_pythainlp
def test_the_change_list_reports_what_was_replaced(spell) -> None:
    fixed, changes = spell.fix("เขาเปนคนดี")
    assert changes == [("เปน", "เป็น")]
    assert fixed == "เขาเป็นคนดี"


@needs_pythainlp
def test_duplicated_leading_vowel_is_normalised(spell) -> None:
    """"เเม่" ที่พิมพ์ด้วย เ สองตัว ต้องกลายเป็น "แม่" ไม่งั้นตัวตัดคำอ่านไม่ออกทั้งคำ"""
    assert spell.fix("เเม่ผมสบายดี")[0] == "แม่ผมสบายดี"


# ──────────────────────────────── 3. โครงพยัญชนะห้ามเปลี่ยน (ด่านกันเดามั่ว)
def test_skeleton_keeps_consonants_and_drops_marks() -> None:
    assert skeleton("เป็น") == skeleton("เปน") == "เปน"
    assert skeleton("โทรศัพท์") == "โทรศพท"          # ั และ ์ หลุดออก
    assert skeleton("ข้าว") == "ขาว"                  # า เป็นอักษรเต็มตัว จึงยังอยู่
    assert skeleton("เสนอ") != skeleton("เปนอ")       # ป → ส คือเปลี่ยนพยัญชนะ


@needs_pythainlp
def test_a_consonant_swap_is_never_accepted(spell) -> None:
    """เศษคำที่ตัวตัดคำสร้างขึ้นต้องไม่ถูกแก้เป็นคำจริงที่ความหมายคนละเรื่อง

    "อากาศเปนอยางไร" ถูกตัดเป็น ...|เปนอ|ยาง|ไร — ถ้ายอมให้เปลี่ยนพยัญชนะ
    "เปนอ" จะกลายเป็น "เสนอ" ซึ่งอ่านแล้วดูน่าเชื่อกว่าคำผิดเดิมเสียอีก
    """
    fixed, changes = spell.fix("วันนี้อากาศเปนอยางไรครับ")
    assert "เสนอ" not in fixed
    assert all(skeleton(a) == skeleton(b) for a, b in changes)


# ────────────────────────────────────────── สิ่งที่ต้องไม่ถูกแตะเลย
@needs_pythainlp
@pytest.mark.parametrize("text", [
    "ราคา 25 บาท",
    "ขอราคาจาก Bangkok Bank หน่อย",
    "https://example.com/a?b=1",
    "ok",
    "25/12/2568",
])
def test_non_thai_text_passes_through(spell, text) -> None:
    assert spell.fix(text) == (text, [])


@needs_pythainlp
def test_words_shorter_than_min_len_are_left_alone() -> None:
    """คำสั้น ๆ เดาผิดง่าย — "ร" เคยถูกแก้เป็น "ระ" ทั้งที่เป็นเศษจากตัวตัดคำ"""
    assert ThaiSpell(min_len=3).fix("ประชากรเทาไหร")[0] == "ประชากรเทาไหร"


@needs_pythainlp
def test_keep_words_are_protected() -> None:
    """คำที่ผู้ใช้สั่งห้ามแก้ต้องรอดแม้พจนานุกรมไม่รู้จัก"""
    word = "เปน"
    assert ThaiSpell().fix(f"เขา{word}คนดี")[0] != f"เขา{word}คนดี"
    assert ThaiSpell(keep=(word,)).fix(f"เขา{word}คนดี")[0] == f"เขา{word}คนดี"


@needs_pythainlp
def test_a_keep_word_the_tokeniser_would_split_is_still_protected() -> None:
    """กันแค่ตอนแก้คำไม่พอ — ต้องยัดเข้าพจนานุกรมตัวตัดคำด้วย

    "เปนดี" ถูกตัดเป็น "เปน"+"ดี" ก่อน แล้ว "เปน" โดนแก้เป็น "เป็น" ทั้งที่ผู้ใช้
    สั่งห้ามแก้ทั้งคำไว้ — รายการ SPELL_KEEP_WORDS จึงไม่ได้กันอะไรเลย
    """
    assert ThaiSpell().fix("เขาเปนดีมาก")[0] == "เขาเป็นดีมาก"      # ไม่ได้สั่งห้าม
    assert ThaiSpell(keep=("เปนดี",)).fix("เขาเปนดีมาก") == ("เขาเปนดีมาก", [])


@needs_pythainlp
def test_a_high_min_freq_refuses_every_correction() -> None:
    strict = ThaiSpell(min_freq=10_000_000_000)
    assert strict.fix("เขาเปนคนดี") == ("เขาเปนคนดี", [])


@needs_pythainlp
def test_text_longer_than_the_cap_is_returned_untouched(spell) -> None:
    from vc.thaispell import MAX_CHARS

    long = "เขาเปนคนดี" * (MAX_CHARS // 10 + 1)
    assert spell.fix(long) == (long, [])


# ───────────────────────────────────── 4. เครื่องที่ไม่มี PyThaiNLP / ของพัง
def test_disabled_is_an_exact_identity() -> None:
    off = ThaiSpell(enabled=False)
    assert off.fix("เขาเปนคนดี") == ("เขาเปนคนดี", [])


def test_missing_pythainlp_degrades_to_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr("vc.thaispell.engine", lambda: None)
    assert ThaiSpell().fix("เขาเปนคนดี") == ("เขาเปนคนดี", [])


def test_a_broken_engine_never_breaks_the_turn(monkeypatch) -> None:
    """ตัวตัดคำระเบิดกลางทาง = คืนข้อความเดิม ไม่ใช่ทำให้ทั้งเทิร์นล้ม"""
    class Boom:
        letters = "ก"
        known: frozenset[str] = frozenset()

        def normalize(self, text: str) -> str:
            raise RuntimeError("พจนานุกรมเสีย")

    monkeypatch.setattr("vc.thaispell.engine", lambda: Boom())
    assert ThaiSpell().fix("เขาเปนคนดี") == ("เขาเปนคนดี", [])


def test_load_failure_is_reported_once(monkeypatch, caplog) -> None:
    """โหลดพจนานุกรมไม่ได้ต้องเตือนครั้งเดียวต่อ process ไม่ใช่ทุกเทิร์น"""
    import vc.thaispell as ts

    def boom(*a, **k):
        raise ImportError("ไม่มี pythainlp")

    monkeypatch.setattr(ts, "_engine", None)
    monkeypatch.setattr(ts, "_failed", False)
    monkeypatch.setattr(ts, "_Engine", boom)
    with caplog.at_level("WARNING"):
        assert ts.engine() is None
        assert ts.engine() is None
    assert sum("ปิดการแก้คำผิด" in r.message for r in caplog.records) == 1


def test_prewarm_does_not_raise_without_pythainlp(monkeypatch) -> None:
    import vc.thaispell as ts

    monkeypatch.setattr(ts, "_engine", None)
    monkeypatch.setattr(ts, "_failed", True)    # ไม่ต้องมีเธรดอีกแล้ว
    ThaiSpell().prewarm()


# ─────────────────── 5. คำตอบของโมเดลก็ถูกแก้ก่อนออกเสียง (TTS_SPELLCHECK)
def _capture(session) -> tuple[list[str], list[tuple]]:
    """ดักข้อความที่ถูกส่งเข้า TTS จริง และ tag ที่ส่งให้ลำโพง"""
    import numpy as np

    said: list[str] = []
    tags: list[tuple] = []

    def synthesize(text: str, sr: int):
        said.append(text)
        return np.zeros(240, np.int16)

    def play(pcm, tag, epoch):
        tags.append(tag)
        return True

    session.api.synthesize = synthesize          # type: ignore[method-assign]
    session.speaker.play = play                  # type: ignore[method-assign]
    return said, tags


def _speak_once(session, text: str) -> None:
    session.enqueue_tts(session.epoch, 0, text)
    session.tts_queue.put(None)                  # ให้ลูปจบเองในเธรดของเทสต์
    session._tts_loop()


@needs_pythainlp
def test_the_reply_is_spellchecked_before_it_is_spoken(web_session) -> None:
    said, _ = _capture(web_session)
    _speak_once(web_session, "เขาเปนคนดี")
    assert said == ["เขาเป็นคนดี"]


@needs_pythainlp
def test_the_screen_and_history_keep_what_the_model_actually_wrote(web_session) -> None:
    """เหมือน TTS_READ_NUMBERS — การแก้คำมีผลกับเสียงเท่านั้น

    tag ที่ส่งให้ลำโพงคือข้อความต้นฉบับ เพราะมันคือสิ่งที่ `_spoken_since()` เอาไป
    ประกอบกลับเป็นประวัติสนทนาเมื่อถูกพูดแทรกกลางคัน
    """
    said, tags = _capture(web_session)
    _speak_once(web_session, "เขาเปนคนดี")
    assert [t[2] for t in tags] == ["เขาเปนคนดี"]
    assert said == ["เขาเป็นคนดี"]


def test_tts_spellcheck_can_be_turned_off(web_session) -> None:
    web_session.cfg.tts_spellcheck = False
    said, _ = _capture(web_session)
    _speak_once(web_session, "เขาเปนคนดี")
    assert said == ["เขาเปนคนดี"]


# ────────────────────────────── 6. ต่อเข้ากับเทิร์นจริง (vc/chat.py)
@needs_pythainlp
def test_the_turn_uses_the_corrected_text(cfg) -> None:
    """ข้อความที่ผ่าน _spellcheck คือข้อความเดียวที่ทุกอย่างปลายน้ำเห็น"""
    from vc.chat import VoiceChat

    chat = VoiceChat(cfg)
    try:
        assert chat._spellcheck(1, "เขาเปนคนดี") == "เขาเป็นคนดี"
        assert chat._spellcheck(1, "") == ""
    finally:
        chat.log.close()
        chat.api.close()


@needs_pythainlp
def test_a_correction_is_logged_with_the_words_it_changed(cfg) -> None:
    from vc.chat import VoiceChat

    chat = VoiceChat(cfg)
    try:
        chat._spellcheck(7, "เขาเปนคนดี")
        events = [e for e in _events(chat) if e["type"] == "spell_fix"]
        assert len(events) == 1, "ไม่ได้บันทึกว่าแก้อะไรไป — ตามย้อนหลังไม่ได้"
        assert events[0]["count"] == 1
        assert "เปน→เป็น" in events[0]["detail"]
    finally:
        chat.log.close()
        chat.api.close()


@needs_pythainlp
def test_nothing_is_logged_when_nothing_changed(cfg) -> None:
    from vc.chat import VoiceChat

    chat = VoiceChat(cfg)
    try:
        chat._spellcheck(1, "เขาเป็นคนดี")
        assert not [e for e in _events(chat) if e["type"] == "spell_fix"]
    finally:
        chat.log.close()
        chat.api.close()


@needs_pythainlp
def test_asr_output_is_corrected_before_it_leaves_the_transcribe_step(cfg) -> None:
    """จุดต่อจริง: ทุกอย่างหลังจากนี้ (เทียบเสียงสะท้อน, ประวัติ, จอ) เห็นข้อความที่แก้แล้ว"""
    import numpy as np

    from vc.chat import VoiceChat

    chat = VoiceChat(cfg)
    try:
        chat.api.transcribe = lambda pcm, sr: "เขาเปนคนดี"   # type: ignore[method-assign]
        assert chat._transcribe_turn(1, np.zeros(160, np.int16)) == "เขาเป็นคนดี"
    finally:
        chat.log.close()
        chat.api.close()


def test_spellcheck_can_be_turned_off_from_config(cfg) -> None:
    from dataclasses import replace

    from vc.chat import VoiceChat

    chat = VoiceChat(replace(cfg, asr_spellcheck=False))
    try:
        assert chat._spellcheck(1, "เขาเปนคนดี") == "เขาเปนคนดี"
    finally:
        chat.log.close()
        chat.api.close()


def _events(chat) -> list[dict]:
    import json

    return [json.loads(line)
            for line in chat.log.jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
