"""ตรวจจับเสียงพูดแบบปรับตัวเอง + กันเสียงลำโพงตัวเองเพื่อให้พูดแทรกได้จริง

หลักการ:
  * ประเมินระดับเสียงรบกวนรอบข้าง (noise floor) แบบ EMA ตอนที่ยังไม่มีใครพูด
  * ขณะ AI พูดออกลำโพง ไมค์จะได้ยินเสียงลำโพงด้วย → ประเมินอัตราการรั่ว
    (leak = rms_ไมค์ / rms_ลำโพง) แล้วยกเพดานขึ้นเป็น leak * rms_ลำโพง * margin
    เสียงคนพูดจริงจะดังทะลุเพดานนี้ ส่วนเสียงสะท้อนของตัวเองจะไม่ทะลุ
  * ต้องมีเสียงต่อเนื่องครบ confirm_ms จึงนับว่าเริ่มพูด (ตอน AI พูดใช้ค่าที่นานกว่า)
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

from .audio import Microphone, Speaker, rms_i16
from .config import Config


class VoiceGate(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        mic: Microphone,
        speaker: Speaker,
        on_speech_start: Callable[[], None],
        on_utterance: Callable[[np.ndarray], None],
        on_level: Callable[[float, float], None] | None = None,
    ):
        super().__init__(name="voice-gate", daemon=True)
        self.cfg = cfg
        self.mic = mic
        self.speaker = speaker
        self.on_speech_start = on_speech_start
        self.on_utterance = on_utterance
        self.on_level = on_level

        self._stop = threading.Event()
        self.paused = threading.Event()      # หยุดฟังชั่วคราว (โหมดพิมพ์)

        fm = cfg.frame_ms
        self._need_idle = max(1, cfg.vad_confirm_ms // fm)
        self._need_play = max(1, cfg.vad_confirm_ms_playback // fm)
        self._need_silence = max(1, cfg.vad_end_silence_ms // fm)
        self._max_frames = max(1, cfg.vad_max_utterance_ms // fm)
        self._min_frames = max(1, cfg.vad_min_utterance_ms // fm)
        self._preroll_frames = max(1, 300 // fm)

        self.noise = 0.004
        self.leak = 0.25
        self._preroll: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []
        self._voiced_run = 0
        self._silence_run = 0
        self.in_speech = False

    # ------------------------------------------------------------------ setup
    def calibrate(self, seconds: float = 1.0) -> float:
        """วัดเสียงรบกวนรอบข้างก่อนเริ่มใช้งาน"""
        n = max(1, int(seconds * 1000 / self.cfg.frame_ms))
        vals: list[float] = []
        for _ in range(n):
            try:
                vals.append(rms_i16(self.mic.frames.get(timeout=1.0)))
            except queue.Empty:
                break
        if vals:
            vals.sort()
            self.noise = max(0.0008, vals[len(vals) // 2])
        return self.noise

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.mic.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.paused.is_set():
                if self.in_speech:
                    self._reset()
                continue
            self._process(frame)

    def _process(self, frame: np.ndarray) -> None:
        level = rms_i16(frame)
        spk = self.speaker.recent_rms() if self.cfg.echo_guard else 0.0
        playing = spk > 0.004

        # noise floor: อัปเดตเฉพาะตอนเงียบจริง ๆ และลำโพงไม่ทำงาน
        if not self.in_speech and not playing and level < self.noise * 3.0 + 0.02:
            self.noise = 0.93 * self.noise + 0.07 * level
            self.noise = max(self.noise, 0.0008)

        threshold = max(self.cfg.vad_abs_threshold, self.noise * self.cfg.vad_noise_mult)
        need = self._need_idle
        if playing:
            # ประเมินอัตรารั่วจากลำโพงเข้าไมค์ (เฉพาะช่วงที่ยังไม่ตัดสินว่ามีคนพูด)
            if not self.in_speech and self._voiced_run == 0:
                obs = min(1.5, level / spk)
                self.leak = 0.88 * self.leak + 0.12 * obs
                self.leak = float(np.clip(self.leak, 0.02, 1.5))
            threshold = max(threshold, self.leak * spk * self.cfg.echo_margin)
            need = self._need_play

        if self.on_level is not None:
            self.on_level(level, threshold)

        if not self.in_speech:
            self._preroll.append(frame)
            if len(self._preroll) > self._preroll_frames:
                self._preroll.pop(0)
            if level > threshold:
                self._voiced_run += 1
                if self._voiced_run >= need:
                    self.in_speech = True
                    self._buf = list(self._preroll)
                    self._preroll = []
                    self._voiced_run = 0
                    self._silence_run = 0
                    self.on_speech_start()
            else:
                self._voiced_run = 0
            return

        # --- กำลังอัดคำพูด ---
        self._buf.append(frame)
        if level > threshold * 0.6:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self._need_silence or len(self._buf) >= self._max_frames:
            speech_frames = len(self._buf) - self._silence_run
            pcm = np.concatenate(self._buf) if self._buf else np.zeros(0, np.int16)
            self._reset()
            if speech_frames >= self._min_frames:
                self.on_utterance(pcm)

    def _reset(self) -> None:
        self.in_speech = False
        self._buf = []
        self._preroll = []
        self._voiced_run = 0
        self._silence_run = 0
