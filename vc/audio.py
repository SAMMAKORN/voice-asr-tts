"""ไมโครโฟน (สตรีมต่อเนื่อง) + ลำโพง (คิวเล่นเสียงที่หยุดกลางทางได้)"""
from __future__ import annotations

import collections
import platform
import queue
import subprocess
import threading
import warnings

import numpy as np

# numpy 2.5 ประกาศเลิกใช้ `arr.shape = ...` ซึ่ง sounddevice 0.5.x เรียกทุกเฟรม
# ถ้าไม่ปิดไว้ กลไก warning ของ Python (รวม regex) จะถูกรันในเธรดเสียงเรียลไทม์
# 50-100 ครั้งต่อวินาที ทำให้เสียงกระตุกและเพิ่มโอกาสค้างตอนปิด stream
warnings.filterwarnings(
    "ignore", message="Setting the shape on a NumPy array",
    category=DeprecationWarning)

import sounddevice as sd  # noqa: E402

CLOSE_TIMEOUT = 2.0     # รอปิด stream ได้นานสุดเท่านี้ (ดู close_stream)


def rms_i16(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def list_devices() -> str:
    return str(sd.query_devices())


def close_stream(stream) -> bool:
    """ปิด stream ในเธรดแยก พร้อม timeout — คืน False ถ้าปิดไม่ลง

    บน macOS มีจังหวะที่ Pa_StopStream ค้างรอ mutex ของ CoreAudio ขณะที่เธรด
    เสียงของ CoreAudio เองก็ค้างรอ mutex อีกตัวอยู่ = deadlock ถาวร ถ้าเรียกจาก
    เธรดหลักตรง ๆ โปรแกรมจะแขวนหลังกด Ctrl+C แล้วโปรเซสไม่ยอมตาย
    (เจอจริง: main thread ค้างใน AudioOutputUnitStop รอ HAL_HardwarePlugIn mutex)

    จึงยิงคำสั่งปิดในเธรด daemon แล้วรอแบบมีเพดานเวลา ถ้าเกินก็ปล่อยทิ้งไป
    ให้เธรดหลักเดินต่อได้ และใช้ abort() แทน stop() เพราะไม่ต้องรอบัฟเฟอร์หมด
    """
    done = threading.Event()

    def worker() -> None:
        try:
            stream.abort(ignore_errors=True)
        finally:
            try:
                stream.close(ignore_errors=True)
            finally:
                done.set()

    threading.Thread(target=worker, name="audio-close", daemon=True).start()
    return done.wait(CLOSE_TIMEOUT)


def mac_input_volume() -> int | None:
    """ระดับ gain ขาเข้าของ macOS (0-100) — คืน None ถ้าอ่านไม่ได้/ไม่ใช่ macOS

    ค่านี้คูณกับเสียงที่ไมค์ส่งเข้ามาตรง ๆ ถ้าถูกหรี่ไว้ต่ำ VAD จะไม่ได้ยิน
    อะไรเลยทั้งที่ไมค์ทำงานปกติ (เบราว์เซอร์ไม่เจอปัญหานี้เพราะมี AGC ของตัวเอง)
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True, text=True, timeout=3.0)
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


class Microphone:
    """อัดเสียงตลอดเวลา ส่งเฟรมขนาด frame_ms เข้าคิวให้ VAD ไปวิเคราะห์"""

    def __init__(self, samplerate: int, frame_ms: int, device: int | None = None):
        self.target_sr = samplerate
        self.frame_ms = frame_ms
        self.device = device
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=250)
        self.native_sr = samplerate
        self.overflows = 0
        self.gain = 1.0          # ขยายเสียงฝั่งซอฟต์แวร์ (ดู set_gain)
        self.clipped = 0
        self._stream: sd.InputStream | None = None

    @property
    def frame_samples(self) -> int:
        return int(self.target_sr * self.frame_ms / 1000)

    def set_gain(self, gain: float) -> float:
        """ตั้งอัตราขยายเสียงขาเข้า (แทน AGC ของเบราว์เซอร์ที่ฝั่งนี้ไม่มี)

        ขยายทั้งเสียงพูดและเสียงรบกวนเท่ากัน อัตราส่วนสัญญาณต่อสัญญาณรบกวน
        จึงไม่เปลี่ยน เกณฑ์ของ VAD ที่ตั้งไว้กับระดับเสียง "ปกติ" เลยกลับมาใช้ได้
        """
        self.gain = float(np.clip(gain, 1.0, 20.0))
        return self.gain

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
        if self.gain != 1.0:
            boosted = frame.astype(np.float32) * self.gain
            if np.abs(boosted).max(initial=0.0) > 32767.0:
                self.clipped += 1
            frame = np.clip(boosted, -32768.0, 32767.0).astype(np.int16)
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            close_stream(stream)


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

    def close(self) -> bool:
        """คืน False ถ้าปิด stream ไม่ลงในเวลาที่กำหนด (ดู close_stream)"""
        self.stop()
        stream, self._stream = self._stream, None
        return close_stream(stream) if stream is not None else True
