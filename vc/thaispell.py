"""ตรวจและแก้คำผิดภาษาไทยของข้อความที่ถอดจากเสียง ก่อนส่งให้โมเดล (PyThaiNLP)

ASR ได้ยินผิดบ่อยกว่าที่คิด — "เป็นอย่างไร" กลับมาเป็น "เปนอยางไร", "โทรศัพท์"
กลับมาเป็น "โทรศัพท" — โมเดลจึงตอบคนละเรื่องหรือลอกคำผิดนั้นไปใช้ในคำตอบต่อ
โมดูลนี้คือด่านเดียวที่ข้อความจาก ASR ผ่านก่อนเข้าเทิร์น (`VoiceChat._transcribe_turn`)

**PyThaiNLP ถูก import แบบ lazy** เหมือน `sounddevice` ใน `vc/audio.py` — เครื่องที่
ไม่ได้ติดตั้งต้องยังรันได้ครบทุกอย่าง เพียงแต่ไม่มีการแก้คำผิด (เตือนครั้งเดียวใน log)
และทุกข้อผิดพลาดระหว่างแก้คำต้องคืนข้อความเดิม ไม่ใช่ทำให้เทิร์นล้ม

## ทำไมไม่เรียก `pythainlp.spell.correct()` ตรง ๆ

สองเหตุผล ทั้งคู่เจอจากการวัดจริง

1. **ช้า** — `correct()` ไล่ผู้สมัครระยะแก้ 2 ครั้ง (edits2) ราวหนึ่งล้านคำต่อหนึ่ง
   คำที่ไม่รู้จัก ตกราว 90 ms/คำ หรือ ~370 ms ต่อประโยคหนึ่ง ซึ่งบวกเข้าไปในเวลา
   ตอบสนองของทุกเทิร์นตรง ๆ · จำกัดที่ระยะแก้ 1 ครั้งเหลือ ~0.2 ms/คำ
2. **มั่ว** — ระยะแก้ 2 ครั้งไกลพอจะเปลี่ยนความหมาย: "นะค่ะ" ถูกแก้เป็น "ค่ะ"
   (หายไปทั้งพยางค์) ส่วนคำผิดที่ ASR ทำจริงเกือบทั้งหมดห่างแค่ 1 ครั้ง
   (ไม้เอกหาย, การันต์หาย, สลับสระ)

ด่านกันแก้มั่วมีห้าชั้น ทุกชั้นมาจากเคสที่พังจริง

* คำสั้นกว่า `min_len` ไม่แตะ — "ร" ถูกแก้เป็น "ระ" ได้ทั้งที่เป็นเศษจากตัวตัดคำ
* คำที่มีอักษรอื่นปนไม่แตะเลย (ตัวเลข, อังกฤษ, วรรค, เครื่องหมาย)
* คำที่พจนานุกรมตัวตัดคำหรือพจนานุกรมสะกดรู้จักอยู่แล้ว = สะกดถูก ไม่ต้องแก้
  รวม **ชื่อและนามสกุลคนไทย** ~22,000 รายการ ไม่งั้น "สมชาย" กลายเป็น "ลมชาย"
  และคำที่ผู้ใช้สั่งห้ามแก้เอง ซึ่งต้องยัดเข้าพจนานุกรมของ *ตัวตัดคำ* ด้วย (`_Engine.trie`)
  ไม่งั้นมันจะถูกหั่นเป็นเศษก่อนแล้วเศษโดนแก้แทน
* **แก้ได้เฉพาะวรรณยุกต์/สระบน-ล่าง/การันต์** — โครงพยัญชนะต้องเหมือนเดิมเป๊ะ
  (`skeleton()`) ชั้นนี้สำคัญที่สุด เพราะคำที่ ASR ได้ยินผิดเกือบทั้งหมดผิดที่รูป
  วรรณยุกต์ ("เปน"→"เป็น", "เพือน"→"เพื่อน", "โทรศัพท"→"โทรศัพท์") ส่วนการเปลี่ยน
  พยัญชนะคือจุดที่ตัวแก้คำเดามั่วเสมอ: เศษคำ "เปนอ" ถูกแก้เป็น "เสนอ" ซึ่งเป็นคำจริง
  ที่ความหมายคนละเรื่อง — อ่านแล้วดูน่าเชื่อกว่าคำผิดเดิมเสียอีก
* คำที่จะแก้ไปหาต้องพบบ่อยอย่างน้อย `min_freq` ครั้ง — "ลมชาย" มีความถี่ 520
  ส่วนคำที่แก้ถูกจริงอยู่ระดับหมื่นถึงสิบล้าน

ผลที่วัดได้จากด่านทั้งห้า: ประโยคไทยที่สะกดถูกอยู่แล้วไม่ถูกแตะเลยสักคำ (ข้อนี้มี
เทสต์คุมอยู่ — ดู `tests/test_thaispell.py`) ส่วนประโยคที่สะกดผิดจะแก้ให้เท่าที่มั่นใจ

ข้อจำกัดที่ยังแก้ไม่ได้: ตัวตัดคำสะดุดตรงคำผิด แล้วกลืนคำข้างเคียงเข้าไปด้วย
("อากาศเปนอยางไร" → "อากาศ|เปนอ|ยาง|ไร") ก้อนที่ออกมาไม่ใช่คำ จึงมักไม่มีคำที่
โครงพยัญชนะตรงกันให้แก้ และปล่อยผ่านไปตามเดิม · คำผิดที่ยืนเป็นคำเดี่ยว ๆ แก้ได้ตามปกติ
"""
from __future__ import annotations

import logging
import threading
import unicodedata
from collections.abc import Iterable

log = logging.getLogger("voicechat.spell")

# ข้อความยาวกว่านี้ไม่ตรวจ — ผลถอดเสียงหนึ่งเทิร์นไม่เคยยาวขนาดนี้ ถ้ายาวแปลว่า
# มีอะไรผิดปกติ และการไล่ทุกคำจะกินเวลาในเส้นทางที่ผู้ใช้รออยู่
MAX_CHARS = 2_000
# เพดานคำที่จำผลไว้ต่อหนึ่ง session — กันหน่วยความจำโตไม่จำกัดในโหมดเว็บที่เปิดค้างยาว
CACHE_MAX = 4_096

_lock = threading.Lock()
_engine: _Engine | None = None
_failed = False


class _Engine:
    """พจนานุกรมทั้งหมดของ PyThaiNLP — โหลดครั้งเดียวต่อ process (ราว 0.4 วินาที)"""

    def __init__(self) -> None:
        # import ในเมธอด ไม่ใช่หัวไฟล์ — เครื่องที่ไม่มี PyThaiNLP ต้อง import
        # โมดูลนี้ (และทั้ง web server) ได้ตามปกติ
        from pythainlp import thai_letters
        from pythainlp.corpus import (thai_family_names, thai_female_names,
                                      thai_male_names, thai_words)
        from pythainlp.spell import NorvigSpellChecker
        from pythainlp.tokenize import word_tokenize
        from pythainlp.util import dict_trie, normalize

        self.letters = thai_letters
        self.checker = NorvigSpellChecker()
        self.tokenize = word_tokenize
        self.normalize = normalize
        self._dict_trie = dict_trie
        self._words = frozenset(thai_words())
        self._tries: dict[frozenset[str], object] = {}
        # คำที่ถือว่า "สะกดถูกแล้ว" — พจนานุกรมตัวตัดคำ (~62,000 คำ) บวกชื่อคนไทย
        self.known: frozenset[str] = (
            frozenset(thai_words()) | frozenset(thai_male_names())
            | frozenset(thai_female_names()) | frozenset(thai_family_names()))

    def trie(self, extra: frozenset[str]) -> object | None:
        """พจนานุกรมตัวตัดคำที่มีคำของผู้ใช้รวมอยู่ด้วย (None = ไม่มีคำเพิ่ม)

        ต้องยัดเข้าพจนานุกรมของ *ตัวตัดคำ* ไม่ใช่แค่กันไว้ตอนแก้คำ ไม่งั้นคำที่ผู้ใช้
        สั่งห้ามแก้จะถูกหั่นเป็นเศษก่อน แล้วเศษนั้นโดนแก้แทน — รายการ
        `SPELL_KEEP_WORDS` จึงไม่ได้กันอะไรเลย · สร้างครั้งเดียวต่อชุดคำ (ราว 0.3
        วินาที) เพราะโหมดเว็บสร้าง session ใหม่ทุกครั้งที่เปิดหน้า
        """
        if not extra:
            return None
        if extra not in self._tries:
            self._tries[extra] = self._dict_trie(self._words | set(extra))
        return self._tries[extra]

    def edits1(self, word: str) -> set[str]:
        """ทุกคำที่ห่างจาก `word` หนึ่งครั้งของการแก้ (ลบ/สลับ/แทน/แทรก)"""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        out = {left + right[1:] for left, right in splits if right}
        out |= {left + right[1] + right[0] + right[2:]
                for left, right in splits if len(right) > 1}
        out |= {left + ch + right[1:]
                for left, right in splits if right for ch in self.letters}
        out |= {left + ch + right for left, right in splits for ch in self.letters}
        out.discard(word)
        return out


def engine() -> _Engine | None:
    """คืนตัวตรวจคำผิด — คืน None ถ้าไม่มี PyThaiNLP หรือโหลดพจนานุกรมไม่สำเร็จ"""
    global _engine, _failed
    if _engine is not None or _failed:
        return _engine
    with _lock:
        if _engine is None and not _failed:
            try:
                _engine = _Engine()
            except Exception as exc:      # noqa: BLE001 — ไม่มีไลบรารี/คลังคำเสีย
                _failed = True            # เตือนครั้งเดียวต่อ process
                log.warning("ปิดการแก้คำผิดภาษาไทย: ใช้ PyThaiNLP ไม่ได้ (%s) — "
                            "ติดตั้งด้วย pip install pythainlp", exc)
    return _engine


def skeleton(word: str) -> str:
    """คำที่ถอดวรรณยุกต์/สระบน-ล่าง/การันต์ออกหมด เหลือแต่โครงพยัญชนะและสระเรียง

    Unicode จัดอักขระกลุ่มนี้เป็น Mn (nonspacing mark) พอดีกับที่ต้องการ:
    ่ ้ ๊ ๋ ็ ์ ั ิ ี ึ ื ุ ู เป็น Mn ส่วน ะ า ำ เ แ โ ใ ไ เป็นอักษรเต็มตัว จึงยังอยู่ในโครง
    """
    return "".join(ch for ch in word if unicodedata.category(ch) != "Mn")


class ThaiSpell:
    """แก้คำผิดภาษาไทยหนึ่งชุดค่า — ถือแคชของ session ไว้ในตัวเอง"""

    def __init__(self, *, enabled: bool = True, min_freq: int = 1_000,
                 min_len: int = 3, keep: Iterable[str] = ()) -> None:
        self.enabled = bool(enabled)
        self.min_freq = max(0, int(min_freq))
        self.min_len = max(2, int(min_len))
        # คำที่ผู้ใช้สั่งห้ามแก้ (ชื่อโครงการ, ยี่ห้อ, ศัพท์เฉพาะ)
        self.keep = frozenset(w for w in (str(k).strip() for k in keep) if w)
        self._cache: dict[str, str] = {}

    def prewarm(self) -> None:
        """โหลดพจนานุกรมล่วงหน้าในเธรดพื้นหลัง

        โหลดตอนใช้จริงครั้งแรกจะบวก ~0.4 วินาทีให้เทิร์นแรกพอดี ซึ่งเป็นเทิร์นที่
        ผู้ใช้รู้สึกถึงความช้าที่สุด · เธรดเป็น daemon และ `engine()` กันเรียกซ้ำเองอยู่แล้ว
        """
        # โหลดแล้ว/โหลดไม่ผ่านแล้ว = ไม่ต้องมีเธรดอีก (โหมดเว็บสร้าง session ใหม่
        # ได้เรื่อย ๆ ทุก session เรียกเมธอดนี้ แต่พจนานุกรมมีชุดเดียวต่อ process)
        if self.enabled and _engine is None and not _failed:
            threading.Thread(target=engine, name="thaispell-warmup",
                             daemon=True).start()

    def fix(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """คืน (ข้อความที่แก้แล้ว, รายการคำที่เปลี่ยน) — คืนของเดิมถ้าแก้ไม่ได้"""
        if not self.enabled or not text or len(text) > MAX_CHARS:
            return text, []
        eng = engine()
        if eng is None:
            return text, []
        try:
            # normalize ก่อน: รวมสระ/วรรณยุกต์ที่ซ้ำหรือสลับลำดับให้เป็นรูปมาตรฐาน
            # ("เเม่" ที่เขียนด้วย เ สองตัว → "แม่") ไม่งั้นตัวตัดคำอ่านไม่ออกทั้งคำ
            normalized = eng.normalize(text)
            changes: list[tuple[str, str]] = []
            out: list[str] = []
            trie = eng.trie(self.keep)
            tokens = (eng.tokenize(normalized, custom_dict=trie) if trie
                      else eng.tokenize(normalized))
            for token in tokens:
                fixed = self._word(eng, token)
                if fixed != token:
                    changes.append((token, fixed))
                out.append(fixed)
            return "".join(out), changes
        except Exception as exc:          # noqa: BLE001 — ห้ามล้มเทิร์นเพราะเรื่องนี้
            log.warning("แก้คำผิดภาษาไทยไม่สำเร็จ ใช้ข้อความเดิมแทน (%r)", exc)
            return text, []

    def _word(self, eng: _Engine, word: str) -> str:
        """แก้คำเดียว — คืนคำเดิมทุกกรณีที่ไม่มั่นใจ"""
        if len(word) < self.min_len or word in self.keep:
            return word
        if not all(ch in eng.letters for ch in word):
            return word                   # มีตัวเลข/อังกฤษ/วรรค/เครื่องหมายปน
        if word in eng.known or eng.checker.freq(word):
            return word                   # สะกดถูกอยู่แล้ว
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        best = word
        shape = skeleton(word)
        candidates = [c for c in eng.checker.known(eng.edits1(word))
                      if skeleton(c) == shape]
        if candidates:
            top = max(candidates, key=eng.checker.freq)
            if eng.checker.freq(top) >= self.min_freq:
                best = top
        if len(self._cache) >= CACHE_MAX:
            self._cache.clear()
        self._cache[word] = best
        return best

