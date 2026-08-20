"""คำทักทายตอนเริ่มระบบมาจาก LLM ใหม่ทุกครั้ง และไม่ซ้ำของเดิม (vc/greeting.py)

สามเรื่องที่ต้องจริงตลอด
1. ยิง LLM ไม่ได้ด้วยเหตุใดก็ตาม ต้องยังทักทายได้ด้วยรายการสำรอง — ห้าม raise
   และห้ามทำให้การเริ่มระบบล้ม
2. "ไม่ซ้ำ" ต้องข้ามรอบการเปิดโปรแกรมได้ จึงต้องมีไฟล์ความจำ และคำที่เคยใช้
   ต้องถูกส่งกลับเข้า prompt จริง ๆ
3. ไฟล์ความจำคือคำพูด — `LOG_TRANSCRIPT=0` แปลว่าห้ามเขียน
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import replace

import pytest

from vc.api import ApiError
from vc.greeting import (ATTEMPTS, GREETINGS_FEMALE, GREETINGS_MALE, MAX_CHARS,
                         RECENT_MAX, GreetingWriter, clean, greeting_text)

pytestmark = pytest.mark.unit

WINDOWS = sys.platform.startswith("win")

# เพศของเสียงมาจากไฟล์อ้างอิงที่โคลนอยู่ ซึ่งไม่ใช่ประเด็นของเทสต์กลุ่มนี้
FALLBACKS = GREETINGS_MALE + GREETINGS_FEMALE


class FakeApi:
    """ApiClient ปลอม — คืนข้อความที่สั่งไว้ทีละก้อน และเก็บ prompt ที่ได้รับ"""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def chat_stream(self, messages, cancel, *a, **k):
        self.prompts.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        for i in range(0, len(reply), 7):      # สตรีมมาทีละก้อนเหมือนของจริง
            yield reply[i:i + 7]


class DeadApi:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def chat_stream(self, messages, cancel, *a, **k):
        raise self.exc
        yield ""       # pragma: no cover — ทำให้เป็น generator


@pytest.fixture
def llm_cfg(cfg):
    """cfg ของชุดเทสต์ปิดคำทักทายจาก LLM ไว้ (ห้ามออกเน็ต) — เปิดเฉพาะที่นี่"""
    return replace(cfg, greet_from_llm=True)


# ───────────────────────────────────────────── ทางปกติ: ได้คำทักทายจาก LLM
def test_the_greeting_comes_from_the_model(llm_cfg) -> None:
    api = FakeApi("สวัสดีตอนบ่ายครับ ผมพร้อมฟังแล้ว พูดแทรกได้ตลอดนะครับ")
    assert GreetingWriter(llm_cfg, api).make() == (
        "สวัสดีตอนบ่ายครับ ผมพร้อมฟังแล้ว พูดแทรกได้ตลอดนะครับ")
    assert len(api.prompts) == 1


def test_the_prompt_carries_the_time_and_the_gender_rule(llm_cfg) -> None:
    api = FakeApi("สวัสดีค่ะ")
    GreetingWriter(replace(llm_cfg, tts_ref_gender="female"), api).make()
    prompt = "\n".join(m["content"] for m in api.prompts[0])
    assert "พูดแทรก" in prompt, "ไม่ได้สั่งให้บอกว่าพูดแทรกได้"
    assert "น." in prompt, "ไม่ได้บอกเวลาปัจจุบันไป คำทักทายจึงเข้ากับช่วงเวลาไม่ได้"


@pytest.mark.parametrize("raw,expect", [
    ('"สวัสดีครับ"', "สวัสดีครับ"),                 # เครื่องหมายคำพูดครอบ
    ("**สวัสดีครับ**", "สวัสดีครับ"),                # Markdown
    ("คำทักทาย: สวัสดีครับ", "สวัสดีครับ"),          # คำนำที่โมเดลชอบแถม
    ("สวัสดีครับ\nพูดแทรกได้เลย", "สวัสดีครับ พูดแทรกได้เลย"),   # ขึ้นบรรทัดใหม่
    ("  สวัสดีครับ  ", "สวัสดีครับ"),
])
def test_the_model_output_is_cleaned_before_it_is_spoken(raw, expect) -> None:
    assert clean(raw) == expect


def test_a_runaway_reply_is_cut_to_the_cap(llm_cfg) -> None:
    api = FakeApi("สวัสดีครับ " * 500)
    made = GreetingWriter(llm_cfg, api).make()
    assert len(made) <= MAX_CHARS
    # ตัดที่ช่องว่าง ไม่ใช่กลางคำ — ภาษาไทยไม่มีช่องว่างระหว่างคำ ครึ่งคำจะถูกอ่านมั่ว
    assert made.endswith("สวัสดีครับ")


# ─────────────────────────────────────────── ล้มเหลว = ต้องยังทักทายได้
@pytest.mark.parametrize("exc", [ApiError("เน็ตล่ม"), RuntimeError("อะไรก็ไม่รู้")])
def test_a_dead_api_falls_back_to_the_offline_list(llm_cfg, exc) -> None:
    assert GreetingWriter(llm_cfg, DeadApi(exc)).make() in FALLBACKS


def test_an_empty_reply_falls_back(llm_cfg) -> None:
    assert GreetingWriter(llm_cfg, FakeApi("", "")).make() in FALLBACKS


def test_no_api_key_means_no_call_at_all(llm_cfg) -> None:
    api = FakeApi("สวัสดีครับ")
    assert GreetingWriter(replace(llm_cfg, api_key=""), api).make() in FALLBACKS
    assert api.prompts == [], "ยิง API ทั้งที่ยังไม่ได้ตั้งค่า key"


def test_the_switch_turns_the_model_off(cfg) -> None:
    api = FakeApi("สวัสดีครับ")
    assert GreetingWriter(cfg, api).make() in FALLBACKS   # cfg ปิดไว้อยู่แล้ว
    assert api.prompts == []


def test_a_cancelled_start_does_not_call_the_model(llm_cfg) -> None:
    """ผู้ใช้ปิดแท็บทันทีที่เปิด — ไม่ต้องเสียเงินยิง LLM เพื่อคำทักทายที่ไม่มีใครฟัง"""
    api = FakeApi("สวัสดีครับ")
    stop = threading.Event()
    stop.set()
    assert GreetingWriter(llm_cfg, api).make(stop) in FALLBACKS
    assert api.prompts == []


# ──────────────────────────────────────────────────── ความจำข้ามรอบ
def test_the_greeting_is_remembered_for_next_time(llm_cfg) -> None:
    writer = GreetingWriter(llm_cfg, FakeApi("สวัสดีตอนเช้าครับ"))
    writer.make()
    assert writer.recent()[-1] == "สวัสดีตอนเช้าครับ"


def test_what_was_used_before_is_sent_back_as_do_not_repeat(llm_cfg) -> None:
    api = FakeApi("อันแรกครับ", "อันที่สองครับ")
    GreetingWriter(llm_cfg, api).make()
    GreetingWriter(llm_cfg, api).make()          # process ใหม่ = writer ตัวใหม่
    assert "อันแรกครับ" in api.prompts[1][-1]["content"], (
        "ไม่ได้ส่งคำทักทายเดิมกลับเข้า prompt — โมเดลจึงไม่มีทางรู้ว่าอะไรซ้ำ")


def test_a_repeated_answer_is_retried_before_it_is_accepted(llm_cfg) -> None:
    """ลองใหม่ก่อน แต่ถ้ายังซ้ำก็ยอมใช้ของโมเดล — รายการสำรองมีแค่สี่แบบ ซ้ำหนักกว่า"""
    api = FakeApi("อันเดิมครับ")
    GreetingWriter(llm_cfg, api).make()
    again = FakeApi(*["อันเดิมครับ"] * ATTEMPTS)
    assert GreetingWriter(llm_cfg, again).make() == "อันเดิมครับ"
    assert len(again.prompts) == ATTEMPTS, "ได้ของซ้ำมาแล้วไม่ได้ลองใหม่เลย"


def test_almost_the_same_greeting_counts_as_repeated(llm_cfg) -> None:
    """ต่างกันคำเดียวก็คือซ้ำสำหรับคนฟัง — โมเดลชอบคืนแบบนี้มากที่สุด"""
    first = "ตีหนึ่งแล้วนะครับ ผมยังตื่นอยู่เลย คุณพูดแทรกขัดจังหวะได้ตลอดนะครับ"
    near = "ตีหนึ่งแล้วนะครับ ผมยังตื่นอยู่เลย คุณก็พูดแทรกขัดจังหวะได้ตลอดนะครับ"
    api = FakeApi(first)
    GreetingWriter(llm_cfg, api).make()
    again = FakeApi(near, "สวัสดีตอนเช้าครับ พูดแทรกได้เลย")
    assert GreetingWriter(llm_cfg, again).make() == "สวัสดีตอนเช้าครับ พูดแทรกได้เลย"


# ต่างกันจริงทุกอัน — ถ้าใช้ "คำทักทายที่ 1/2/3" ตัวตรวจซ้ำแบบคล้ายกัน (SIMILAR)
# จะตัดสินว่าซ้ำกันหมด แล้วเทสต์นี้จะวัดเรื่องอื่นแทนเพดานความจำ
SEVENTEEN = ("สวัสดี", "หวัดดี", "ดีจ้า", "ยินดีต้อนรับ", "อรุณสวัสดิ์",
             "ราตรีสวัสดิ์", "เฮลโหล", "มาแล้วนะ", "พร้อมแล้ว", "เริ่มกันเลย",
             "ว่าไง", "ทักทายจ้า", "สบายดีไหม", "คิดถึงจัง", "มีอะไรให้ช่วย",
             "ถามมาได้เลย", "โย่")


def test_the_memory_never_grows_past_its_cap(llm_cfg) -> None:
    rounds = RECENT_MAX + 5
    assert len(SEVENTEEN) >= rounds
    api = FakeApi(*SEVENTEEN[:rounds])
    for _ in range(rounds):
        GreetingWriter(llm_cfg, api).make()
    assert len(GreetingWriter(llm_cfg, api).recent()) == RECENT_MAX


def test_a_corrupt_memory_file_is_treated_as_empty(llm_cfg) -> None:
    writer = GreetingWriter(llm_cfg, FakeApi("สวัสดีครับ"))
    writer.path.parent.mkdir(parents=True, exist_ok=True)
    writer.path.write_text("{ไม่ใช่ json", encoding="utf-8")
    assert writer.recent() == []
    assert writer.make() == "สวัสดีครับ"        # ยังทำงานต่อได้ตามปกติ


@pytest.mark.skipif(WINDOWS, reason="สิทธิ์แบบ POSIX ใช้ไม่ได้บน Windows")
def test_the_memory_file_is_private(llm_cfg) -> None:
    writer = GreetingWriter(llm_cfg, FakeApi("สวัสดีครับ"))
    writer.make()
    assert writer.path.stat().st_mode & 0o777 == 0o600
    assert json.loads(writer.path.read_text(encoding="utf-8")) == ["สวัสดีครับ"]


def test_nothing_is_written_when_transcripts_are_off(llm_cfg) -> None:
    """ไฟล์ความจำก็คือคำพูด — LOG_TRANSCRIPT=0 ต้องไม่มีอะไรลงดิสก์เลย"""
    writer = GreetingWriter(replace(llm_cfg, log_transcript=False),
                            FakeApi("สวัสดีครับ"))
    assert writer.make() == "สวัสดีครับ"
    assert not writer.path.exists()


# ──────────────────────────────────────────── รายการสำรอง (ไม่ต้องต่อเน็ต)
def test_the_offline_list_avoids_what_was_just_used() -> None:
    used = list(GREETINGS_MALE[:3])
    assert greeting_text("male", used) == GREETINGS_MALE[3]


def test_the_offline_list_still_answers_when_everything_was_used() -> None:
    assert greeting_text("male", GREETINGS_MALE) in FALLBACKS


def test_the_female_voice_gets_female_wording() -> None:
    assert greeting_text("female") in GREETINGS_FEMALE


# ───────────────────────────────────────────────── ต่อเข้ากับเทิร์นจริง
def test_the_turn_speaks_what_the_writer_produced(web_session) -> None:
    web_session.cfg.tts_enabled = False        # ไม่มีเธรด TTS ในเทสต์นี้
    web_session.greeter = GreetingWriter(replace(web_session.cfg, greet_from_llm=True),
                                         FakeApi("สวัสดีตอนค่ำครับ"))
    web_session.greet()
    assert web_session.messages[-1]["content"] == "สวัสดีตอนค่ำครับ"
    shown = [p["text"] for p in web_session.out.of("delta")]
    assert "สวัสดีตอนค่ำครับ" in shown, "คำทักทายไม่ได้ขึ้นหน้าจอ"


def test_a_session_that_closed_while_thinking_stays_silent(web_session) -> None:
    """ปิดแท็บระหว่างแต่งคำทักทาย — ห้ามต่อคิว TTS ทิ้งไว้ให้ตัวนับค้าง (AC-2.2)"""
    class Slow(FakeApi):
        def chat_stream(self, messages, cancel, *a, **k):
            web_session.running.clear()          # ปิด session กลางคัน
            yield "สวัสดีครับ"

    web_session.greeter = GreetingWriter(replace(web_session.cfg, greet_from_llm=True),
                                         Slow("ไม่ได้ใช้"))
    web_session.greet()
    assert web_session._inflight == 0
    assert web_session.tts_queue.empty()
