"""P2-12 — เสียงของเทิร์นที่ถูกพูดแทรกไปแล้วต้องไม่หลุดออกลำโพงอีก

บั๊กเดิม: `voice_chat.py` เช็ค `epoch != self.epoch` แล้วค่อยเรียก `speaker.play()`
เป็นคนละบรรทัด เธรด VAD สอด `interrupt()` เข้ามาระหว่างสองบรรทัดนั้นได้พอดี
ตอนนี้การเช็คกับการต่อคิวอยู่ใน lock เดียวกันของ Speaker เอง
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from vc.audio import Speaker

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
CHUNK = np.ones(2400, np.int16)


def test_play_is_rejected_when_epoch_moved_on() -> None:
    spk = Speaker(24000)
    spk.set_epoch(7)
    assert spk.play(CHUNK, tag=(7, 1, "ก"), epoch=7) is True
    assert spk.play(CHUNK, tag=(6, 1, "เก่า"), epoch=6) is False
    spk.set_epoch(8)
    assert spk.play(CHUNK, tag=(7, 2, "ค้างจากเทิร์นก่อน"), epoch=7) is False
    assert spk.pending() is False, "set_epoch ต้องล้างเสียงของเทิร์นก่อนทิ้ง"


def test_stop_closes_the_gate_until_a_new_epoch_opens_it() -> None:
    spk = Speaker(24000)
    spk.set_epoch(3)
    spk.play(CHUNK, tag=(3, 1, "ก"), epoch=3)
    spk.stop()
    assert spk.pending() is False
    assert spk.play(CHUNK, tag=(3, 2, "ที่สังเคราะห์เสร็จช้า"), epoch=3) is False, (
        "เสียงที่สังเคราะห์เสร็จหลังถูกพูดแทรกต้องเข้าคิวไม่ได้")
    spk.set_epoch(4)
    assert spk.play(CHUNK, tag=(4, 1, "เทิร์นใหม่"), epoch=4) is True


def test_play_without_epoch_stays_backward_compatible() -> None:
    """ผู้เรียกที่ไม่ผูกกับเทิร์น (เทสต์/เสียงเตือน) ยังใช้ play() แบบเดิมได้"""
    spk = Speaker(24000)
    assert spk.play(CHUNK, tag=("ไม่ผูกเทิร์น",)) is True
    assert spk.pending() is True


# ────────────────────────────────────────────────── ยิงสลับกัน 1,000 รอบ (AC-12.1)
def test_no_stale_audio_survives_a_thousand_interleavings() -> None:
    spk = Speaker(24000)
    rounds = 1000
    leaked: list[tuple[int, int]] = []
    done = threading.Event()

    def tts_worker() -> None:
        """เลียนแบบเธรด TTS: สังเคราะห์เสร็จแล้วยิงเข้าคิวเรื่อย ๆ"""
        while not done.is_set():
            epoch = spk.epoch
            if epoch is None:
                continue
            spk.play(CHUNK, tag=(epoch, 0, "เสียง"), epoch=epoch)

    worker = threading.Thread(target=tts_worker, daemon=True)
    worker.start()
    try:
        for n in range(1, rounds + 1):
            spk.set_epoch(n)
            time.sleep(0)                    # ปล่อยให้เธรด TTS ได้ยิงแทรกจริง
            spk.stop()                       # เธรด VAD พูดแทรกตรงนี้
            # ประตูปิดแล้ว: ห้ามมีเสียงของ epoch ไหนโผล่เข้าคิวได้อีก
            with spk._lock:                  # noqa: SLF001 - ตรวจสถานะภายในโดยตั้งใจ
                stale = [t for _, t in spk._queue if isinstance(t, tuple)]
            if stale:
                leaked.append((n, len(stale)))
    finally:
        done.set()
        worker.join(2.0)

    assert leaked == [], f"เสียงของเทิร์นเก่าหลุดเข้าคิวหลัง stop(): {leaked[:5]}"


def test_lock_does_not_add_noticeable_latency() -> None:
    """AC-12.2 — ค่า overhead ต่อ chunk ต้องไม่เกิน 5ms"""
    spk = Speaker(24000)
    spk.set_epoch(1)
    t0 = time.perf_counter()
    for i in range(500):
        spk.play(CHUNK, tag=(1, i, "ก"), epoch=1)
    per_chunk_ms = (time.perf_counter() - t0) / 500 * 1000
    assert per_chunk_ms < 5.0, f"play() ช้าเกินไป ({per_chunk_ms:.3f} ms/chunk)"


# ───────────────────────────────────────────────────── ไม่มี pattern เดิมหลงเหลือ
def test_no_check_then_play_pattern_left() -> None:
    """AC-12.3 — ต้องไม่เหลือ "เช็ค epoch แล้วค่อย play" นอก lock"""
    # คลาส VoiceChat ย้ายจาก voice_chat.py ไป vc/chat.py แล้ว (P3-21) — เงื่อนไข
    # ที่ตรวจข้างล่างเหมือนเดิมทุกตัวอักษร เปลี่ยนเฉพาะที่อยู่ของไฟล์
    src = (ROOT / "vc" / "chat.py").read_text(encoding="utf-8")
    calls = [line.strip() for line in src.splitlines() if "speaker.play(" in line]
    assert calls, "ไม่พบการเรียก speaker.play เลย — เทสต์นี้ล้าสมัยแล้ว"
    for line in calls:
        assert re.search(r"epoch=epoch", line), (
            f"เรียก play โดยไม่ส่ง epoch ให้ตรวจใน lock: {line}")

    loop = src[src.index("def _tts_loop"):src.index("def _tts_pending")]
    after_synth = loop[loop.index("api.synthesize"):]
    assert "!= self.epoch" not in after_synth, (
        "ยังเช็ค epoch เองหลังสังเคราะห์เสียงก่อนเรียก play (ช่องแทรกเดิม)")
