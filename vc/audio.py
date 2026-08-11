"""ไมโครโฟน (สตรีมต่อเนื่อง) + ลำโพง (คิวเล่นเสียงที่หยุดกลางทางได้)"""
from __future__ import annotations

import collections
import queue
import threading

import numpy as np
import sounddevice as sd


def rms_i16(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def list_devices() -> str:
    return str(sd.query_devices())


class Microphone:
    """อัดเสียงตลอดเวลา ส่งเฟรมขนาด frame_ms เข้าคิวให้ VAD ไปวิเคราะห์"""

    def __init__(self, samplerate: int, frame_ms: int, device: int | None = None):
        self.target_sr = samplerate
        self.frame_ms = frame_ms
        self.device = device
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=250)
        self.native_sr = samplerate
        self.overflows = 0
        self._stream: sd.InputStream | None = None

    @property
    def frame_samples(self) -> int:
        return int(self.target_sr * self.frame_ms / 1000)

    def start(self) -> int:
        last_err: Exception | None = None
        for sr in (self.target_sr, 48000, 44100, 0):
            try:
                if sr == 0:  # ปล่อยให้อุปกรณ์เลือกอัตราของตัวเอง
                    info = sd.query_devices(self.device if self.device is not None
                                            else sd.default.device[0], "input")
                    sr = int(info["default_samplerate"])
                block = int(sr * self.frame_ms / 1000)
                stream = sd.InputStream(
                    samplerate=sr, blocksize=block, device=self.device,
                    channels=1, dtype="int16", callback=self._callback,
                )
                stream.start()
                self._stream, self.native_sr = stream, sr
                return sr
            except Exception as exc:  # ลองอัตราถัดไป
                last_err = exc
        raise RuntimeError(f"เปิดไมโครโฟนไม่ได้: {last_err}")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        frame = indata[:, 0].copy()
        if self.native_sr != self.target_sr:
            n = self.frame_samples
            if self.native_sr % self.target_sr == 0:
                frame = frame[:: self.native_sr // self.target_sr][:n]
            else:
                src = np.linspace(0.0, 1.0, frame.size, endpoint=False)
                dst = np.linspace(0.0, 1.0, n, endpoint=False)
                frame = np.interp(dst, src, frame.astype(np.float32)).astype(np.int16)
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class Speaker:
    """เล่นเสียงจากคิว: play() ต่อคิว, stop() ตัดจบทันที (ใช้เวลาถูกพูดขัด)

    เก็บ RMS ของเสียงที่เพิ่งเล่นไว้ ให้ VAD ใช้ประเมินเสียงลำโพงที่รั่วเข้าไมค์
    """

    def __init__(self, samplerate: int, device: int | None = None):
        self.src_sr = samplerate      # อัตราสุ่มของเสียงที่ TTS ส่งมา
        self.sr = samplerate          # อัตราสุ่มที่อุปกรณ์เปิดได้จริง
        self.device = device
        self._lock = threading.Lock()
        self._queue: collections.deque[tuple[np.ndarray, object]] = collections.deque()
        self._cur: np.ndarray | None = None
        self._cur_tag: object = None
        self._pos = 0
        self._out_rms: collections.deque[float] = collections.deque([0.0] * 10, maxlen=10)
        self.finished_tags: list[object] = []
        self._stream: sd.OutputStream | None = None

    def start(self) -> int:
        last_err: Exception | None = None
        for sr in (self.src_sr, 48000, 44100, 22050, 16000):
            try:
                stream = sd.OutputStream(
                    samplerate=sr, blocksize=int(sr * 0.02), device=self.device,
                    channels=1, dtype="int16", callback=self._callback,
                )
                stream.start()
                self._stream, self.sr = stream, sr
                return sr
            except Exception as exc:  # ลองอัตราถัดไป
                last_err = exc
        raise RuntimeError(f"เปิดลำโพงไม่ได้: {last_err}")

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames:
                if self._cur is None:
                    if not self._queue:
                        break
                    self._cur, self._cur_tag = self._queue.popleft()
                    self._pos = 0
                take = min(frames - filled, self._cur.size - self._pos)
                out[filled:filled + take] = self._cur[self._pos:self._pos + take]
                self._pos += take
                filled += take
                if self._pos >= self._cur.size:
                    if self._cur_tag is not None:
                        self.finished_tags.append(self._cur_tag)
                    self._cur, self._cur_tag = None, None
        if filled < frames:
            out[filled:] = 0
        self._out_rms.append(rms_i16(out[:filled]) if filled else 0.0)

    def play(self, pcm: np.ndarray, tag: object = None) -> None:
        if pcm.size == 0:
            return
        if self.sr != self.src_sr:
            from .api import resample_i16
            pcm = resample_i16(pcm, self.src_sr, self.sr)
        with self._lock:
            self._queue.append((np.ascontiguousarray(pcm, dtype=np.int16), tag))

    def stop(self) -> None:
        with self._lock:
            self._queue.clear()
            self._cur, self._cur_tag = None, None
            self._pos = 0

    def pending(self) -> bool:
        with self._lock:
            return self._cur is not None or bool(self._queue)

    def queued_seconds(self) -> float:
        with self._lock:
            n = sum(a.size for a, _ in self._queue)
            if self._cur is not None:
                n += self._cur.size - self._pos
        return n / self.sr

    def recent_rms(self) -> float:
        """RMS สูงสุดใน ~200ms ที่ผ่านมา (ครอบ latency ของเสียงที่วนกลับเข้าไมค์)"""
        return max(self._out_rms) if self._out_rms else 0.0

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
