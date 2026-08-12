"""ตัวเชื่อมระหว่างตรรกะเดิม (vc/*) กับเบราว์เซอร์

ฝั่งเทอร์มินัลใช้ไมค์/ลำโพงของเครื่องผ่าน sounddevice ส่วนฝั่งเว็บย้ายอุปกรณ์
ไปอยู่บนเบราว์เซอร์ ไฟล์นี้จึงทำ 3 คลาสที่หน้าตาเหมือน Microphone / Speaker /
Console เดิมทุกประการ แต่รับ-ส่งข้อมูลผ่าน WebSocket แทน
ทำให้ VoiceGate และ VoiceChat เดิมนำมาใช้ต่อได้โดยไม่ต้องแก้ตรรกะ barge-in เลย
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time

import numpy as np

Outgoing = tuple[str, object]      # ("json", dict) | ("bin", bytes)


class Outbox:
    """คิวข้อความขาออก — เธรดของ session ใส่เข้ามา, ลูป asyncio ดึงไปส่ง"""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.queue: asyncio.Queue[Outgoing | None] = asyncio.Queue()
        self._closed = False

    def _put(self, item: Outgoing | None) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:      # ลูปปิดไปแล้ว (client หลุด)
            self._closed = True

    def json(self, type: str, **fields) -> None:
        fields["type"] = type
        self._put(("json", fields))

    def binary(self, data: bytes) -> None:
        self._put(("bin", data))

    def close(self) -> None:
        self._put(None)
        self._closed = True


class WebMic:
    """ปลายทางของเฟรมเสียง 16 kHz / 20 ms ที่เบราว์เซอร์สตรีมเข้ามา

    หน้าตาเหมือน vc.audio.Microphone เท่าที่ VoiceGate ต้องใช้ (คือ .frames)
    """

    def __init__(self, samplerate: int, frame_ms: int, maxsize: int = 250):
        self.target_sr = samplerate
        self.frame_ms = frame_ms
        self.native_sr = samplerate
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.received = 0
        self.started = threading.Event()
        self._carry = np.zeros(0, np.int16)

    @property
    def frame_samples(self) -> int:
        return int(self.target_sr * self.frame_ms / 1000)

    def feed(self, pcm: np.ndarray) -> None:
        """แบ่งเสียงที่รับมาเป็นเฟรมขนาดคงที่ก่อนส่งให้ VAD

        เบราว์เซอร์อาจส่งมาไม่ตรงขนาดเป๊ะ จึงกันเศษไว้ต่อกับก้อนถัดไป
        """
        self.received += pcm.size
        if not self.started.is_set():
            self.started.set()
        buf = np.concatenate([self._carry, pcm]) if self._carry.size else pcm
        n = self.frame_samples
        i = 0
        while buf.size - i >= n:
            try:
                self.frames.put_nowait(buf[i:i + n])
            except queue.Full:
                self.dropped += 1
            i += n
        self._carry = buf[i:].copy()

    def stop(self) -> None:
        pass


class WebSpeaker:
    """ลำโพงที่อยู่บนเบราว์เซอร์

    ส่ง PCM ไปให้ client เล่น แล้วรอ client รายงานกลับว่าเล่นก้อนไหนจบแล้ว
    (ต้องรู้ให้แม่นว่าผู้ใช้ได้ยินอะไรไปบ้าง ตอนถูกพูดขัดจะได้บันทึกถูก)

    หน้าตาเหมือน vc.audio.Speaker เท่าที่ VoiceChat/VoiceGate ต้องใช้:
    play / stop / set_epoch / pending / recent_rms / finished_tags

    มีประตูกัน epoch เหมือนกับ vc.audio.Speaker (P2-12) — การเช็ค epoch กับการ
    ส่งเสียงออก WebSocket อยู่ใน lock เดียวกัน เสียงของเทิร์นที่ถูกพูดแทรกไปแล้ว
    จึงไม่มีทางหลุดออกไปถึงเบราว์เซอร์
    """

    RMS_TTL = 0.35        # ถ้า client เงียบหายเกินนี้ ถือว่าไม่ได้เล่นเสียงอยู่
    LOST_GRACE = 5.0      # ไม่มีรายงานว่าเล่นจบเกินความยาวเสียง + เท่านี้ = ถือว่าหาย

    def __init__(self, out: Outbox, samplerate: int):
        self.out = out
        self.src_sr = samplerate
        self.sr = samplerate
        self.finished_tags: list[object] = []
        self._lock = threading.Lock()
        # (epoch, seq) -> (tag, เวลาที่ถือว่าเล่นจบแน่ ๆ แม้ client เงียบหาย)
        self._pending: dict[tuple[int, int], tuple[object, float]] = {}
        self._rms = 0.0
        self._rms_at = 0.0
        self._epoch: int | None = None      # None = ปิดประตู

    def start(self) -> int:
        return self.sr

    # ------------------------------------------------------ ประตูกันเสียงข้ามเทิร์น
    def set_epoch(self, epoch: int) -> None:
        with self._lock:
            self._epoch = int(epoch)
            self._pending.clear()

    @property
    def epoch(self) -> int | None:
        with self._lock:
            return self._epoch

    # ------------------------------------------------------------ ส่งเสียงออก
    def play(self, pcm: np.ndarray, tag: object = None,
             epoch: int | None = None) -> bool:
        if pcm.size == 0:
            return False
        tag_epoch, seq = 0, 0
        if isinstance(tag, tuple) and len(tag) >= 2:
            tag_epoch, seq = int(tag[0]), int(tag[1])
        deadline = time.monotonic() + pcm.size / self.sr + self.LOST_GRACE
        body = np.ascontiguousarray(pcm, "<i2").tobytes()
        header = np.array([tag_epoch, seq], dtype="<u4").tobytes()
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                return False
            self._pending[(tag_epoch, seq)] = (tag, deadline)
            # ส่งออกในล็อกด้วย เพื่อไม่ให้ stop() ที่มาพร้อมกันแซงลำดับข้อความ
            self.out.binary(header + body)
            return True

    def stop(self) -> None:
        """ตัดเสียงที่ค้างอยู่ทั้งหมดทันที (ใช้ตอนถูกพูดขัด) + ปิดประตู"""
        with self._lock:
            self._epoch = None
            self._pending.clear()
            self._rms = 0.0
            self.out.json("stop")

    def pending(self) -> bool:
        """ยังมีเสียงรออยู่ไหม — ทิ้งก้อนที่ client ไม่รายงานกลับ จะได้ไม่ค้างตลอดกาล"""
        now = time.monotonic()
        with self._lock:
            for key, (_, deadline) in list(self._pending.items()):
                if now > deadline:
                    del self._pending[key]
            return bool(self._pending)

    # ------------------------------------------------- รายงานกลับจากเบราว์เซอร์
    def note_played(self, epoch: int, seq: int) -> None:
        with self._lock:
            item = self._pending.pop((epoch, seq), None)
        if item is not None:
            self.finished_tags.append(item[0])

    def note_level(self, rms: float) -> None:
        with self._lock:
            self._rms = max(0.0, min(1.0, float(rms)))
            self._rms_at = time.monotonic()

    def recent_rms(self) -> float:
        with self._lock:
            if time.monotonic() - self._rms_at > self.RMS_TTL:
                return 0.0
            return self._rms

    def queued_seconds(self) -> float:
        return 0.0

    def close(self) -> None:
        with self._lock:
            self._pending.clear()


class WebConsole:
    """แทน vc.ui.Console — แทนที่จะวาดบนเทอร์มินัล ก็ยิง event ไปขึ้นบนหน้าเว็บ"""

    def __init__(self, out: Outbox):
        self.out = out
        self.color = False
        self._lock = threading.RLock()
        self._open_role = ""
        self._status = ""

    def _c(self, code: str, text: str) -> str:
        return text

    @property
    def width(self) -> int:
        return 100

    # ------------------------------------------------------------------ สถานะ
    def status(self, icon: str, text: str, color: str = "") -> None:
        with self._lock:
            line = f"{icon}{text}"
            if line == self._status:
                return
            self._status = line
            self.out.json("status", icon=icon, text=text)

    def clear_status(self) -> None:
        with self._lock:
            if not self._status:
                return
            self._status = ""
            self.out.json("status", icon="", text="")

    # ----------------------------------------------------------------- caption
    def begin(self, label: str, color: str, role: str = "") -> None:
        with self._lock:
            if self._open_role:
                self.end()
            self._open_role = role or "assistant"
            self.out.json("begin", role=self._open_role, label=label.strip())

    def write(self, text: str) -> None:
        with self._lock:
            if not self._open_role:
                return
            self.out.json("delta", text=text)

    def end(self, suffix: str = "") -> None:
        with self._lock:
            if not self._open_role:
                return
            self._open_role = ""
            self.out.json("end", interrupted=bool(suffix), note=suffix.strip())

    # -------------------------------------------------------------- ข้อความอื่น
    def line(self, text: str = "") -> None:
        if text.strip():
            self.out.json("log", level="info", text=text.strip())

    def note(self, text: str) -> None:
        self.line(text)

    def warn(self, text: str) -> None:
        self.out.json("log", level="warn", text=text.strip())

    def error(self, text: str) -> None:
        self.out.json("log", level="error", text=text.strip())

    def sources(self, items: list[dict]) -> None:
        if items:
            self.out.json("sources", items=items[:6])
