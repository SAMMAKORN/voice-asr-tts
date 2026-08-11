#!/usr/bin/env python3
"""ทดสอบตรรกะที่ไม่ต้องใช้ฮาร์ดแวร์: VAD, การพูดแทรก, การตัดประโยค

รัน: python3 tests/test_offline.py
"""
from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vc.chunker import SentenceChunker, clean_for_tts  # noqa: E402
from vc.config import Config  # noqa: E402
from vc.vad import VoiceGate  # noqa: E402

SR = 16000
FRAME = 20
N = int(SR * FRAME / 1000)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeMic:
    def __init__(self) -> None:
        self.frames: queue.Queue[np.ndarray] = queue.Queue()


class FakeSpeaker:
    def __init__(self, rms: float = 0.0) -> None:
        self.rms = rms

    def recent_rms(self) -> float:
        return self.rms


def tone(level: float, n: int = N, seed: int = 0) -> np.ndarray:
    """เสียงคล้ายเสียงพูด: noise + ฮาร์มอนิก, RMS ≈ level"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 420 * t)
           + 0.3 * rng.standard_normal(n))
    sig = sig / np.sqrt(np.mean(sig ** 2)) * level
    return np.clip(sig * 32768, -32768, 32767).astype(np.int16)


def build(cfg_overrides: dict | None = None, speaker_rms: float = 0.0):
    cfg = Config(mic_sr=SR, frame_ms=FRAME, **(cfg_overrides or {}))
    mic, spk = FakeMic(), FakeSpeaker(speaker_rms)
    events: list[str] = []
    utts: list[np.ndarray] = []
    gate = VoiceGate(cfg, mic, spk,  # type: ignore[arg-type]
                     on_speech_start=lambda: events.append("start"),
                     on_utterance=utts.append)
    gate.noise = 0.002
    return gate, spk, events, utts


def feed(gate: VoiceGate, frames: list[np.ndarray]) -> None:
    for f in frames:
        gate._process(f)


def silence(n_frames: int, level: float = 0.0015) -> list[np.ndarray]:
    return [tone(level, seed=i + 500) for i in range(n_frames)]


def speech(n_frames: int, level: float = 0.09) -> list[np.ndarray]:
    return [tone(level, seed=i) for i in range(n_frames)]


# ------------------------------------------------------------------ tests
def test_basic_utterance() -> None:
    print("\n[1] จับคำพูดปกติ แล้วตัดจบเมื่อเงียบ")
    gate, _, events, utts = build()
    feed(gate, silence(25))
    check("ยังไม่ตรวจพบเสียงพูดตอนเงียบ", events == [] and utts == [])
    feed(gate, speech(60))                    # พูด 1.2 วินาที
    check("ตรวจพบว่าเริ่มพูด", events == ["start"])
    check("ยังไม่ตัดจบระหว่างพูด", utts == [])
    feed(gate, silence(40))                   # เงียบ 0.8 วินาที > 650ms
    check("ตัดจบเมื่อเงียบครบเกณฑ์", len(utts) == 1,
          f"{utts[0].size / SR:.2f}s" if utts else "")
    if utts:
        dur = utts[0].size / SR
        check("ความยาวรวม preroll สมเหตุสมผล (1.2-2.5s)", 1.2 <= dur <= 2.5,
              f"{dur:.2f}s")


def test_short_noise_ignored() -> None:
    print("\n[2] เสียงสั้น ๆ (เคาะโต๊ะ) ต้องไม่กลายเป็นคำพูด")
    gate, _, events, utts = build()
    feed(gate, silence(20))
    feed(gate, speech(4))                     # 80ms < min utterance
    feed(gate, silence(45))
    check("ไม่ส่งไปถอดเสียง", utts == [], f"utterances={len(utts)}")


def test_barge_in_with_echo_guard() -> None:
    print("\n[3] กันเสียงลำโพงตัวเอง แต่ยังพูดแทรกได้")
    # ลำโพงกำลังดัง 0.20; ไมค์ได้ยินเสียงสะท้อนที่ 15% → ไม่ควรนับเป็นคนพูด
    gate, spk, events, utts = build(speaker_rms=0.20)
    gate.leak = 0.15
    feed(gate, [tone(0.20 * 0.15, seed=i + 90) for i in range(60)])
    check("เสียงสะท้อนจากลำโพงไม่ทำให้เกิด barge-in", events == [],
          f"leak={gate.leak:.3f}")

    # คนพูดจริงดังกว่าเสียงสะท้อนหลายเท่า → ต้องขัดได้
    feed(gate, speech(30, level=0.20))
    check("เสียงคนพูดจริงทำให้เกิด barge-in", events == ["start"])
    spk.rms = 0.0                              # ลำโพงถูกสั่งหยุดเมื่อถูกขัด
    feed(gate, silence(40))
    check("ได้คำพูดของผู้ใช้ไปถอดเสียง", len(utts) == 1)


def test_barge_in_headphones() -> None:
    print("\n[4] โหมดหูฟัง (ปิด echo guard) ขัดได้ไวขึ้น")
    gate, _, events, utts = build({"echo_guard": False}, speaker_rms=0.30)
    feed(gate, silence(15))
    n_before = 0
    for i, f in enumerate(speech(20, level=0.05)):
        gate._process(f)
        if events and not n_before:
            n_before = i + 1
    check("ขัดได้ภายใน ~120ms", 0 < n_before <= 7, f"{n_before * FRAME}ms")


def test_max_duration() -> None:
    print("\n[5] พูดยาวเกินเพดานต้องตัดส่งเอง")
    gate, _, events, utts = build({"vad_max_utterance_ms": 1000})
    feed(gate, silence(15))
    feed(gate, speech(120))                   # 2.4s > เพดาน 1s
    check("ตัดส่งอัตโนมัติ", len(utts) >= 1, f"utterances={len(utts)}")


def test_chunker() -> None:
    print("\n[6] ตัดข้อความสตรีมเป็นก้อนให้ TTS")
    ch = SentenceChunker()
    text = ("สวัสดีครับ วันนี้อากาศที่กรุงเทพร้อนมาก อุณหภูมิประมาณ 35 องศา "
            "แนะนำให้พกร่มไปด้วยนะครับ เพราะช่วงบ่ายอาจมีฝนฟ้าคะนองบางแห่ง")
    out: list[str] = []
    for i in range(0, len(text), 3):          # จำลองการสตรีมทีละ 3 ตัวอักษร
        out += ch.feed(text[i:i + 3])
    rest = ch.flush()
    if rest:
        out.append(rest)
    check("ได้หลายก้อน", len(out) >= 3, f"{len(out)} ก้อน")
    check("ก้อนแรกสั้นเพื่อเริ่มพูดเร็ว", len(out[0]) <= 40, repr(out[0]))
    joined = "".join(out).replace(" ", "")
    check("ไม่มีข้อความหาย", joined == text.replace(" ", ""))
    check("ทุกก้อนไม่ยาวเกินเพดาน", all(len(c) <= 170 for c in out))

    md = "**สรุป**: ทานข้าวต้ม `แนะนำ` [ลิงก์](https://x.com)\n- ข้อแรก"
    cleaned = clean_for_tts(md)
    check("ล้าง Markdown ก่อนอ่านออกเสียง",
          "*" not in cleaned and "`" not in cleaned and "https" not in cleaned,
          repr(cleaned))


def main() -> int:
    print("ทดสอบตรรกะแบบไม่ใช้ฮาร์ดแวร์")
    test_basic_utterance()
    test_short_noise_ignored()
    test_barge_in_with_echo_guard()
    test_barge_in_headphones()
    test_max_duration()
    test_chunker()
    print()
    if FAILURES:
        print(f"ล้มเหลว {len(FAILURES)} ข้อ: " + ", ".join(FAILURES))
        return 1
    print("ผ่านทั้งหมด ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
