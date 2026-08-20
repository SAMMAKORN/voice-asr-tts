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
import math
import threading
import unicodedata
from collections.abc import Iterable

log = logging.getLogger("voicechat.spell")

# ข้อความยาวกว่านี้ไม่ตรวจ — ผลถอดเสียงหนึ่งเทิร์นไม่เคยยาวขนาดนี้ ถ้ายาวแปลว่า
# มีอะไรผิดปกติ และการไล่ทุกคำจะกินเวลาในเส้นทางที่ผู้ใช้รออยู่
MAX_CHARS = 2_000
# เพดานคำที่จำผลไว้ต่อหนึ่ง session — กันหน่วยความจำโตไม่จำกัดในโหมดเว็บที่เปิดค้างยาว
CACHE_MAX = 4_096

# --- การแบ่งคำใหม่รอบสอง (ดูหัวข้อ "ตัวตัดคำสะดุด" ในคำอธิบายไฟล์) ---
WINDOW_CHARS = 20     # ช่วงที่ยอมแบ่งใหม่ยาวสุดกี่ตัวอักษร
WINDOW_TOKENS = 4     # กวาดก้อนถัดไปเข้ามาในช่วงได้กี่ก้อน
PIECE_MAX = 12        # คำไทยหนึ่งคำยาวสุดที่ยอมให้เป็นชิ้นหนึ่งของการแบ่งใหม่
RESEGMENT_MAX = 12    # ลองแบ่งใหม่ได้กี่ช่วงต่อการเรียกหนึ่งครั้ง (กันเวลาบานปลาย)
# --- การแก้คำผิดบนสตรีม (StreamSpell) ---
STREAM_MIN = 24       # สั้นกว่านี้ยังไม่ปล่อย รอให้มีบริบทพอจะตัดคำถูกก่อน
STREAM_KEEP = 2       # กันก้อนท้ายไว้กี่ก้อน (ก้อนสุดท้ายมักเป็นคำที่ยังพิมพ์ไม่จบ)
# ค่าปรับของการ "แก้หนึ่งคำ" ในหน่วย log ความถี่ — การแบ่งที่ต้องแก้คำจะชนะการแบ่ง
# ที่ไม่ต้องแก้ ก็ต่อเมื่อผลลัพธ์เป็นคำที่พบบ่อยกว่ากันราว e^6 ≈ 400 เท่า
FIX_PENALTY = 6.0
# ค่าปรับต่อ "หนึ่งชิ้น" — เอนไปทางคำยาวไม่กี่คำ แทนที่จะแตกเป็นเศษสั้น ๆ หลายชิ้น
PIECE_PENALTY = 2.0
# ค่าปรับเพิ่มของชิ้นที่สั้นกว่า `min_len` — เศษสองตัวอักษรที่พจนานุกรมรับรองว่า
# เป็นคำ ("นอ" ความถี่ 40 ล้าน, "แล" 96 ล้าน) มีอยู่เกลื่อน ถ้าไม่ปรับ การแบ่งที่
# ถูกต้อง ("เป็น|อย่าง") จะแพ้เศษที่อ่านไม่รู้เรื่อง ("เป้|นอ|ยาง") เสมอ · ไม่ห้าม
# ขาดเพราะคำสองตัวอักษรจริงก็มี ("ผม", "ไป", "มา") แค่ต้องไม่มีทางเลือกที่ยาวกว่า
SHORT_PENALTY = 8.0

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


def marks(word: str) -> int:
    """จำนวนวรรณยุกต์/สระบน-ล่าง/การันต์ในคำ (อักขระกลุ่ม Mn)"""
    return sum(1 for ch in word if unicodedata.category(ch) == "Mn")


def align(span: str, pieces: tuple[str, ...]) -> list[tuple[str, str]]:
    """จับคู่ข้อความเดิมกับชิ้นที่แบ่งใหม่ เพื่อบันทึกว่าคำไหนถูกเปลี่ยนเป็นอะไร

    บันทึกทั้งช่วง ("เปนคนดี→เป็นคนดี") อ่านยากเวลาไล่ดูว่าตัวแก้คำไปแตะอะไรบ้าง
    ตัดหัวท้ายที่เหมือนกันแบบตัวอักษรก็ไม่ได้ เพราะภาษาไทยมีสระนำหน้าพยัญชนะ
    ("เปน" จะถูกตัดเหลือ "ป") · ใช้ `skeleton()` เป็นไม้บรรทัดแทน — โครงพยัญชนะ
    ของทั้งช่วงถูกบังคับให้เท่าเดิมอยู่แล้ว ความยาวโครงของแต่ละชิ้นจึงชี้ได้ว่า
    ชิ้นนั้นกินข้อความเดิมไปถึงไหน
    """
    out: list[tuple[str, str]] = []
    pos = 0
    for piece in pieces:
        need = len(skeleton(piece))
        take = 0
        while pos + take < len(span) and len(skeleton(span[pos:pos + take])) < need:
            take += 1
        # วรรณยุกต์/สระบน-ล่างที่ห้อยท้ายเป็นของชิ้นนี้ ไม่ใช่ของชิ้นถัดไป
        while (pos + take < len(span)
               and unicodedata.category(span[pos + take]) == "Mn"):
            take += 1
        src, pos = span[pos:pos + take], pos + take
        if src != piece:
            out.append((src, piece))
    return out


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
        # `normalize()` ตัดวรรคหัวท้ายทิ้งด้วย ซึ่งกลืนวรรคหายเวลาข้อความถูกป้อน
        # มาทีละก้อน (StreamSpell) — เก็บไว้เองแล้วต่อกลับตอนคืนค่า
        lead = text[:len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        core = text.strip()
        if not core:
            return text, []
        try:
            # normalize ก่อน: รวมสระ/วรรณยุกต์ที่ซ้ำหรือสลับลำดับให้เป็นรูปมาตรฐาน
            # ("เเม่" ที่เขียนด้วย เ สองตัว → "แม่") ไม่งั้นตัวตัดคำอ่านไม่ออกทั้งคำ
            normalized = eng.normalize(core)
            out: list[str] = []
            trie = eng.trie(self.keep)
            tokens = (eng.tokenize(normalized, custom_dict=trie) if trie
                      else eng.tokenize(normalized))
            # รอบแรก: ซ่อมช่วงที่ตัวตัดคำหั่นผิดเพราะสะดุดคำผิด แล้วค่อยไล่ทีละคำ
            tokens, changes = self._resplit(eng, list(tokens))
            for token in tokens:
                fixed = self._word(eng, token)
                if fixed != token:
                    changes.append((token, fixed))
                out.append(fixed)
            return lead + "".join(out) + trail, changes
        except Exception as exc:          # noqa: BLE001 — ห้ามล้มเทิร์นเพราะเรื่องนี้
            log.warning("แก้คำผิดภาษาไทยไม่สำเร็จ ใช้ข้อความเดิมแทน (%r)", exc)
            return text, []

    def _word(self, eng: _Engine, word: str, min_len: int = 0) -> str:
        """แก้คำเดียว — คืนคำเดิมทุกกรณีที่ไม่มั่นใจ

        `min_len` ทับเพดานความยาวของ instance เฉพาะตอนแบ่งคำใหม่ ซึ่งมีบริบท
        ทั้งช่วงคุมอยู่แล้ว ("นี" → "นี้" ต้องแก้ได้ แม้ SPELL_MIN_LEN จะเป็น 3)
        """
        if len(word) < (min_len or self.min_len) or word in self.keep:
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
        # โครงพยัญชนะเท่าเดิม **และมาร์กต้องไม่ลดลง** — ด่านหลังมาจากของจริง:
        # skeleton() มองข้ามวรรณยุกต์/สระบน-ล่างทั้งหมด การ "ลบ" มาร์กทิ้งจึงผ่าน
        # ด่านแรกเสมอ แล้ว "เพือน" ถูกแก้เป็น "เพอน" (ความถี่พอผ่านเกณฑ์ด้วย)
        # แทนที่จะเป็น "เพื่อน" · คำผิดที่ ASR ทำคือมาร์ก *หาย* หรือสลับ ไม่ใช่มาร์กเกิน
        need = marks(word)
        candidates = [c for c in eng.checker.known(eng.edits1(word))
                      if skeleton(c) == shape and marks(c) >= need]
        if candidates:
            top = max(candidates, key=eng.checker.freq)
            if eng.checker.freq(top) >= self.min_freq:
                best = top
        if len(self._cache) >= CACHE_MAX:
            self._cache.clear()
        self._cache[word] = best
        return best

    # ------------------------------------------------------ แบ่งคำใหม่รอบสอง
    def _known(self, eng: _Engine, word: str) -> bool:
        return word in eng.known or bool(eng.checker.freq(word))

    def _piece(self, eng: _Engine, span: str) -> tuple[str, int, float] | None:
        """`span` เป็นคำอะไรได้บ้าง — คืน (คำ, จำนวนครั้งที่แก้, log ความถี่)

        คืน None เมื่อ span ไม่ใช่คำและแก้ให้เป็นคำไม่ได้ · ชิ้นตัวอักษรเดียวต้อง
        เป็นคำที่พบบ่อยจริงเท่านั้น ไม่งั้นการแบ่งจะแตกเป็นตัวอักษรเรียงกันได้
        """
        short = SHORT_PENALTY if len(span) < self.min_len else 0.0
        if self._known(eng, span):
            return span, 0, math.log(eng.checker.freq(span) + 1) - short
        # ชิ้นที่ "ต้องแก้" ยอมสั้นได้ถึงสองตัว เพราะบริบททั้งช่วงคุมอยู่แล้ว
        # ("วันนีอากาศ" → "วัน|นี้|อากาศ" ต้องแก้ได้ แม้ SPELL_MIN_LEN จะเป็น 3)
        if len(span) < 2 or span in self.keep:
            return None
        fixed = self._word(eng, span, min_len=2)
        if fixed == span:
            return None
        return fixed, 1, math.log(eng.checker.freq(fixed) + 1) - short

    def _resegment(self, eng: _Engine,
                   span: str) -> tuple[tuple[str, ...], float] | None:
        """แบ่ง `span` ใหม่ให้ทุกชิ้นเป็นคำจริง โดยแก้คำผิดได้ระหว่างทาง

        เลือกการแบ่งที่ผลรวม log ความถี่สูงสุด หักค่าปรับต่อการแก้หนึ่งครั้ง —
        การแบ่งที่ไม่ต้องแก้อะไรเลยจึงชนะเสมอ ยกเว้นการแก้จะให้คำที่พบบ่อยกว่ามาก
        คืน None เมื่อแบ่งไม่ได้ทั้งช่วง หรือแบ่งได้แต่ผลลัพธ์เหมือนเดิม
        """
        n = len(span)
        # best[k] = (คะแนน, ชิ้นที่แบ่งได้ของ span[:k])
        best: list[tuple[float, tuple[str, ...]] | None] = [None] * (n + 1)
        best[0] = (0.0, ())
        for k in range(1, n + 1):
            for size in range(1, min(PIECE_MAX, k) + 1):
                prev = best[k - size]
                if prev is None:
                    continue
                got = self._piece(eng, span[k - size:k])
                if got is None:
                    continue
                word, fixes, weight = got
                score = (prev[0] + weight - PIECE_PENALTY
                         - FIX_PENALTY * fixes)
                if best[k] is None or score > best[k][0]:
                    best[k] = (score, prev[1] + (word,))
        done = best[n]
        if done is None:
            return None
        joined = "".join(done[1])
        if joined == span or skeleton(joined) != skeleton(span):
            return None      # ไม่เปลี่ยนอะไร หรือเปลี่ยนโครงพยัญชนะ = ไม่เอา
        return done[1], done[0] / n      # ต่อหนึ่งตัวอักษร เทียบข้ามช่วงได้

    def _suspect(self, eng: _Engine, token: str) -> bool:
        """ก้อนนี้น่าจะเป็นเศษที่ตัวตัดคำหั่นผิดไหม"""
        return (len(token) >= 2 and token not in self.keep
                and all(ch in eng.letters for ch in token)
                and not self._known(eng, token))

    def _resplit(self, eng: _Engine,
                 tokens: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """ซ่อมช่วงที่ตัวตัดคำหั่นผิด — ด่านที่ทำให้ตัวแก้คำได้ทำงานจริง

        ตัวตัดคำสะดุดตรงคำผิดแล้วกลืนตัวแรกของคำถัดไปเข้ามาด้วย
        ("อากาศเปนอยางไร" → "อากาศ|เปนอ|ยาง|ไร") ก้อน "เปนอ" ไม่ใช่คำ และไม่มีคำ
        ไหนที่โครงพยัญชนะตรงกัน การแก้ทีละก้อนจึงเงียบสนิททั้งประโยค — เคสนี้คือ
        คำผิดจาก ASR ส่วนใหญ่ ไม่ใช่ข้อยกเว้น จึงต้องต่อก้อนกลับเข้าด้วยกันแล้ว
        แบ่งใหม่ทั้งช่วง ("เปนอยาง" → "เป็น|อย่าง")

        ขยายช่วงทีละก้อนแล้วหยุดที่ช่วงแรกที่แบ่งได้ — ช่วงสั้นที่สุดที่อธิบายได้
        คือช่วงที่เสี่ยงน้อยที่สุด
        """
        out: list[str] = []
        changes: list[tuple[str, str]] = []
        budget = RESEGMENT_MAX
        i = 0
        while i < len(tokens):
            if budget <= 0 or not self._suspect(eng, tokens[i]):
                out.append(tokens[i])
                i += 1
                continue
            budget -= 1
            hit: tuple[int, tuple[str, ...]] | None = None
            score = 0.0
            for j in range(i + 1, min(i + WINDOW_TOKENS, len(tokens)) + 1):
                span = "".join(tokens[i:j])
                if len(span) > WINDOW_CHARS:
                    break
                got = self._resegment(eng, span)
                # เทียบทุกช่วงแล้วเอาช่วงที่ "อธิบายได้ดีที่สุดต่อหนึ่งตัวอักษร"
                # หยุดที่ช่วงแรกที่แบ่งได้ไม่พอ: "เปนอ" แบ่งเป็น "เป้|นอ" ได้ก็จริง
                # แต่ช่วงที่กว้างอีกหน่อย ("เปนอยาง") ให้ "เป็น|อย่าง" ซึ่งดีกว่ามาก
                if got is not None and (hit is None or got[1] > score):
                    hit, score = (j, got[0]), got[1]
            if hit is None:
                out.append(tokens[i])
                i += 1
            else:
                changes.extend(align("".join(tokens[i:hit[0]]), hit[1]))
                out.extend(hit[1])
                i = hit[0]
        return out, changes


class StreamSpell:
    """แก้คำผิดบนข้อความที่ไหลมาทีละชิ้น โดยไม่ตัดกลางคำ

    คำตอบของโมเดลมาเป็น token ทีละไม่กี่ตัวอักษร ส่งเข้า `ThaiSpell.fix()` ดิบ ๆ
    ไม่ได้ — ครึ่งคำที่ยังพิมพ์ไม่จบจะถูกมองเป็นคำผิดแล้วโดนแก้เป็นคำอื่น
    (แย่กว่าไม่แก้เลย) · จึงสะสมไว้ก่อน แล้วปล่อยเฉพาะส่วนที่ตัวตัดคำยืนยันว่า
    จบคำแล้วจริง เก็บ `STREAM_KEEP` ก้อนท้ายไว้รอชิ้นถัดไปเสมอ

    ผลข้างเคียงที่ยอมรับ: ช่วงที่คร่อมรอยต่อจะไม่ถูกแบ่งคำใหม่ (`_resplit`)
    เพราะมองไม่เห็นข้อความอีกฝั่ง — แลกกับการที่หน้าจอยังไหลตามคำตอบทันที
    """

    def __init__(self, spell: ThaiSpell) -> None:
        self.spell = spell
        self._buf = ""

    @property
    def enabled(self) -> bool:
        return self.spell.enabled

    def feed(self, text: str) -> str:
        """รับชิ้นใหม่ คืนส่วนที่แก้เสร็จแล้วและปล่อยออกได้ (อาจเป็นค่าว่าง)"""
        if not self.spell.enabled:
            return text
        self._buf += text
        if len(self._buf) < STREAM_MIN:
            return ""
        eng = engine()
        if eng is None:               # ไม่มี PyThaiNLP = ปล่อยผ่านตามเดิม
            out, self._buf = self._buf, ""
            return out
        try:
            tokens = list(eng.tokenize(self._buf))
        except Exception:             # noqa: BLE001 — ห้ามล้มกลางคำตอบ
            out, self._buf = self._buf, ""
            return out
        if len(tokens) <= STREAM_KEEP:
            return ""
        head = "".join(tokens[:-STREAM_KEEP])
        self._buf = self._buf[len(head):]
        return self.spell.fix(head)[0]

    def flush(self) -> str:
        """ปล่อยส่วนที่ค้างทั้งหมด — เรียกตอนสตรีมจบหรือถูกพูดแทรก"""
        rest, self._buf = self._buf, ""
        if not rest or not self.spell.enabled:
            return rest
        return self.spell.fix(rest)[0]
