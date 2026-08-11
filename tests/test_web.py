#!/usr/bin/env python3
"""ทดสอบสะพานเชื่อมฝั่งเว็บโดยไม่ใช้เบราว์เซอร์และไม่เรียก API

ตรวจว่า WebMic / WebSpeaker / WebConsole ทำตัวเหมือน Microphone / Speaker /
Console ของเดิมพอที่ VoiceGate และ VoiceChat จะทำงานต่อได้ถูกต้อง
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vc.config import Config                                    # noqa: E402
from vc.vad import VoiceGate                                    # noqa: E402
from web.bridge import Outbox, WebConsole, WebMic, WebSpeaker   # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✓ {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


class FakeOutbox(Outbox):
    """เก็บข้อความไว้ในลิสต์แทนการส่งออก WebSocket จริง"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self._closed = False

    def _put(self, item):  # type: ignore[override]
        if item is not None and not self._closed:
            self.sent.append(item)

    def close(self) -> None:
        self._closed = True

    def kinds(self) -> list[str]:
        return [p.get("type") for k, p in self.sent if k == "json"]  # type: ignore[union-attr]

    def of(self, kind: str) -> list[dict]:
        return [p for k, p in self.sent
                if k == "json" and p.get("type") == kind]  # type: ignore[union-attr]

    def blobs(self) -> list[bytes]:
        return [p for k, p in self.sent if k == "bin"]  # type: ignore[misc]


def tone(ms: int, sr: int, amp: float, freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(sr * ms / 1000)) / sr
    wave = np.sin(2 * np.pi * freq * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (wave * amp * 32767).astype(np.int16)


def silence(ms: int, sr: int, amp: float = 0.0006) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.normal(0, amp, int(sr * ms / 1000)) * 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────── 1
def test_mic_framing() -> None:
    print("\n[1] WebMic ตัดเสียงที่เบราว์เซอร์ส่งมาเป็นเฟรมขนาดคงที่")
    mic = WebMic(16000, 20)          # 320 ตัวอย่างต่อเฟรม
    check("ยังไม่มีเฟรมก่อนได้รับข้อมูล", mic.frames.qsize() == 0)

    mic.feed(np.zeros(500, np.int16))         # ไม่ลงตัว เหลือเศษ 180
    check("ได้ 1 เฟรมจาก 500 ตัวอย่าง", mic.frames.qsize() == 1,
          f"qsize={mic.frames.qsize()}")
    check("เก็บเศษไว้ต่อก้อนถัดไป", mic._carry.size == 180, f"carry={mic._carry.size}")

    mic.feed(np.zeros(140, np.int16))         # 180 + 140 = 320 พอดี
    check("เศษ + ก้อนใหม่ = อีก 1 เฟรม", mic.frames.qsize() == 2)
    check("ทุกเฟรมยาว 320", all(mic.frames.get().size == 320 for _ in range(2)))
    check("started ถูกตั้งหลังได้เสียงก้อนแรก", mic.started.is_set())

    mic2 = WebMic(16000, 20, maxsize=3)
    mic2.feed(np.zeros(320 * 10, np.int16))
    check("คิวเต็มแล้วทิ้งเฟรมส่วนเกินแทนที่จะบล็อก",
          mic2.frames.qsize() == 3 and mic2.dropped == 7, f"dropped={mic2.dropped}")


# ─────────────────────────────────────────────────────────── 2
def test_speaker() -> None:
    print("\n[2] WebSpeaker: ส่งเสียงไปเบราว์เซอร์แล้วรอรายงานว่าเล่นจบ")
    out = FakeOutbox()
    spk = WebSpeaker(out, 24000)

    pcm = tone(200, 24000, 0.3)
    spk.play(pcm, tag=(4, 1, "สวัสดี"))
    spk.play(pcm, tag=(4, 2, "ครับ"))

    blobs = out.blobs()
    check("ส่งเสียงออกไป 2 ก้อน", len(blobs) == 2)
    head = np.frombuffer(blobs[0][:8], dtype="<u4")
    check("หัวข้อมูลบอก epoch/seq ถูก", tuple(head) == (4, 1), f"head={tuple(head)}")
    body = np.frombuffer(blobs[0][8:], dtype="<i2")
    check("เนื้อเสียงตรงกับที่ส่ง", body.size == pcm.size and np.array_equal(body, pcm))

    check("ยังถือว่ามีเสียงค้างอยู่", spk.pending() is True)
    spk.note_played(4, 1)
    check("รายงานเล่นจบก้อนแรกแล้วยังเหลือก้อนสอง", spk.pending() is True)
    spk.note_played(4, 2)
    check("รายงานครบแล้วถือว่าเล่นจบหมด", spk.pending() is False)
    check("finished_tags เรียงตามลำดับที่เล่นจริง",
          [t[1] for t in spk.finished_tags] == [1, 2])

    # ถูกพูดขัด: ก้อนที่ยังไม่ได้เล่นต้องหายไป และต้องสั่ง client หยุด
    spk.play(pcm, tag=(4, 3, "ที่ยังไม่ได้ยิน"))
    spk.stop()
    check("stop() ล้างเสียงที่ค้าง", spk.pending() is False)
    check("stop() สั่งให้เบราว์เซอร์หยุดเล่น", out.kinds().count("stop") == 1)
    check("ก้อนที่ถูกตัดไม่ถูกนับว่าผู้ใช้ได้ยิน", len(spk.finished_tags) == 2)


# ─────────────────────────────────────────────────────────── 3
def test_speaker_guards() -> None:
    print("\n[3] WebSpeaker กันค้างเมื่อเบราว์เซอร์เงียบหาย")
    out = FakeOutbox()
    spk = WebSpeaker(out, 24000)
    spk.LOST_GRACE = 0.15
    spk.play(tone(100, 24000, 0.3), tag=(1, 1, "ก"))
    check("แรก ๆ ยังรออยู่", spk.pending() is True)
    time.sleep(0.35)
    check("เกินเวลาแล้วปล่อยผ่าน ไม่ค้างตลอดกาล", spk.pending() is False)

    spk2 = WebSpeaker(out, 24000)
    check("ยังไม่มีรายงานความดัง → ถือว่าไม่ได้เล่นเสียง", spk2.recent_rms() == 0.0)
    spk2.note_level(0.42)
    check("รับค่าความดังจากเบราว์เซอร์ได้", abs(spk2.recent_rms() - 0.42) < 1e-6)
    spk2.RMS_TTL = 0.05
    time.sleep(0.12)
    check("ค่าความดังหมดอายุเมื่อ client เงียบไป", spk2.recent_rms() == 0.0)


# ─────────────────────────────────────────────────────────── 4
def test_console() -> None:
    print("\n[4] WebConsole แปลง caption เป็น event ให้หน้าเว็บ")
    out = FakeOutbox()
    c = WebConsole(out)

    c.begin("🧑 คุณ", "", role="user")
    c.write("อากาศวันนี้")
    c.end()
    c.begin("🤖 AI ", "", role="assistant")
    c.write("ร้อนมาก")
    c.end("  ⟨ถูกพูดขัด⟩")

    check("ลำดับ event ถูกต้อง",
          out.kinds() == ["begin", "delta", "end", "begin", "delta", "end"],
          str(out.kinds()))
    check("แยก role ผู้พูดได้", [b["role"] for b in out.of("begin")] == ["user", "assistant"])
    ends = out.of("end")
    check("รอบแรกไม่ถูกขัด", ends[0]["interrupted"] is False)
    check("รอบสองระบุว่าถูกพูดขัด", ends[1]["interrupted"] is True)

    c.status("💭", "กำลังคิด...")
    c.status("💭", "กำลังคิด...")      # ซ้ำ ต้องไม่ส่งอีก
    check("สถานะซ้ำไม่ยิงซ้ำ", len(out.of("status")) == 1)
    c.clear_status()
    check("ล้างสถานะแล้วส่งค่าว่าง", out.of("status")[-1]["text"] == "")

    c.warn("ระวัง")
    c.error("พัง")
    levels = [m["level"] for m in out.of("log")]
    check("แยกระดับ log ได้", levels == ["warn", "error"], str(levels))


# ─────────────────────────────────────────────────────────── 5
def test_gate_with_web_devices() -> None:
    print("\n[5] VoiceGate เดิมทำงานกับไมค์/ลำโพงฝั่งเว็บได้")
    cfg = Config(vad_confirm_ms=120, vad_end_silence_ms=400,
                 vad_min_utterance_ms=300, echo_guard=False)
    out = FakeOutbox()
    mic = WebMic(cfg.mic_sr, cfg.frame_ms)
    spk = WebSpeaker(out, cfg.speaker_sr)

    started = threading.Event()
    got: list[np.ndarray] = []
    gate = VoiceGate(cfg, mic, spk,
                     on_speech_start=started.set,
                     on_utterance=got.append)

    mic.feed(silence(1200, cfg.mic_sr))
    noise = gate.calibrate(1.0)
    check("วัดเสียงรบกวนจากเฟรมที่เบราว์เซอร์ส่งมาได้", noise < 0.005, f"noise={noise:.5f}")

    gate.start()
    mic.feed(silence(200, cfg.mic_sr))
    mic.feed(tone(900, cfg.mic_sr, 0.25))
    mic.feed(silence(700, cfg.mic_sr))

    deadline = time.time() + 5.0
    while time.time() < deadline and not got:
        time.sleep(0.02)
    gate.stop()

    check("ตรวจพบว่าผู้ใช้เริ่มพูด", started.is_set())
    check("ตัดจบแล้วส่งคำพูดไปถอดเสียง", len(got) == 1, f"utterances={len(got)}")
    if got:
        ms = got[0].size / cfg.mic_sr * 1000
        check("ความยาวคำพูดสมเหตุสมผล (900ms + preroll)", 900 <= ms <= 1700,
              f"{ms:.0f}ms")


# ─────────────────────────────────────────────────────────── 6
def test_outbox_threading() -> None:
    print("\n[6] Outbox ส่งข้ามเธรดเข้าลูป asyncio ได้")

    async def main() -> list:
        loop = asyncio.get_running_loop()
        out = Outbox(loop)
        threading.Thread(target=lambda: (out.json("ready", n=1),
                                         out.binary(b"\x01\x02"),
                                         out.close())).start()
        items = []
        while True:
            item = await asyncio.wait_for(out.queue.get(), timeout=3.0)
            if item is None:
                return items
            items.append(item)

    items = asyncio.run(main())
    check("ได้ทั้ง json และ binary ตามลำดับ",
          [k for k, _ in items] == ["json", "bin"], str([k for k, _ in items]))
    check("เนื้อหา json ถูกต้อง", items[0][1] == {"n": 1, "type": "ready"})
    check("ปิดคิวแล้วหยุดวน", True)


def main() -> int:
    test_mic_framing()
    test_speaker()
    test_speaker_guards()
    test_console()
    test_gate_with_web_devices()
    test_outbox_threading()
    print()
    if FAIL:
        print(f"ล้มเหลว {FAIL} ข้อ (ผ่าน {PASS}) ✗")
        return 1
    print(f"ผ่านทั้งหมด {PASS} ข้อ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
