#!/usr/bin/env python3
"""คุยโต้ตอบกับ AI ด้วยเสียงแบบเรียลไทม์ (ไทย) — ตัวเรียกฝั่งบรรทัดคำสั่ง

  เสียงเข้า → ASR → LLM (สตรีม) → TTS → เสียงออก
  พูดแทรกได้ตลอดเวลา · แสดง caption ทันที · บันทึกบทสนทนาลง logs/

ไฟล์นี้ทำแค่สามอย่าง: อ่าน argument, ประกอบค่าตั้ง, แล้วสั่งงานแกนกลางใน `vc/`
ตัวคลาส `VoiceChat` ย้ายไปอยู่ที่ `vc/chat.py` แล้ว (P3-21) เพื่อให้โหมดเว็บไม่ต้อง
import สคริปต์ CLI — ชื่อเดิมยัง import จากที่นี่ได้เหมือนเดิมทั้งหมด

ใช้ค่า API/โมเดลจาก .env ทั้งหมด
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from vc.audio import Microphone, list_devices, mac_input_volume
from vc.chat import (CUT_MARK, EVENTS_MAXSIZE, FINDINGS_HEADER, GREETING,
                     LOW_INPUT_VOLUME, NO_ANSWER_MARK, SEARCH_FILLER,
                     TTS_WAIT_POLL, TTS_WAIT_TIMEOUT, VoiceChat, phase_log)
from vc.config import Config, load_config
from vc.options import RuntimeOptions
from vc.selftest import run_selftest
from vc.ui import Console
from vc.vad import VoiceGate

# ชื่อที่เคยอยู่ในไฟล์นี้ — คงไว้ให้โค้ด/เทสต์เดิมที่ `from voice_chat import ...` ใช้ได้
__all__ = [
    "VoiceChat", "GREETING", "SEARCH_FILLER", "CUT_MARK", "NO_ANSWER_MARK",
    "FINDINGS_HEADER", "LOW_INPUT_VOLUME", "TTS_WAIT_TIMEOUT", "TTS_WAIT_POLL",
    "EVENTS_MAXSIZE", "phase_log", "mic_check", "selftest", "main",
]


# ----------------------------------------------------------------- mic check
def mic_check(cfg: Config, seconds: float = 20.0) -> int:
    """มิเตอร์เสียงเข้าแบบสด — ดูว่าเสียงพูดขึ้นถึงเกณฑ์ที่ VAD ใช้จริงหรือเปล่า

    ใช้ Microphone + VoiceGate ตัวเดียวกับตอนใช้งานจริง ไม่ได้จำลองขึ้นใหม่
    """
    console = Console()
    vol = mac_input_volume()
    console.line()
    console.line("  ตรวจไมโครโฟน — พูดตามปกติจนกว่าจะครบเวลา")
    if vol is not None:
        console.note(f"  input volume ของเครื่อง: {vol}%"
                     + ("   ← ต่ำเกินไป" if vol < LOW_INPUT_VOLUME else ""))

    mic = Microphone(cfg.mic_sr, cfg.frame_ms, cfg.input_device)
    native = mic.start()
    if native != cfg.mic_sr:
        console.note(f"  ไมค์ทำงานที่ {native} Hz → แปลงเป็น {cfg.mic_sr} Hz")

    class _Silent:      # ลำโพงหลอก: ตรวจไมค์อย่างเดียว ไม่ต้องกันเสียงสะท้อน
        def recent_rms(self) -> float:
            return 0.0

    heard: list[int] = []
    live = {"level": 0.0, "threshold": cfg.vad_abs_threshold}

    def on_level(level: float, threshold: float) -> None:
        live["level"] = level
        live["threshold"] = threshold

    gate = VoiceGate(cfg, mic, _Silent(),          # type: ignore[arg-type]
                     on_speech_start=lambda: None,
                     on_utterance=lambda pcm: heard.append(pcm.size),
                     on_level=on_level)
    console.note("  กำลังวัดเสียงรบกวน อยู่เงียบ ๆ 1.5 วินาที...")
    noise = gate.calibrate(1.5)
    gain = mic.set_gain(cfg.mic_gain or min(8.0, cfg.mic_target_noise / max(noise, 1e-6)))
    gate.noise = noise * gain
    threshold = max(cfg.vad_abs_threshold, gate.noise * cfg.vad_noise_mult)
    console.note(f"  เสียงรบกวน {noise:.5f} · ขยาย {gain:.1f} เท่า → {gate.noise:.5f} · "
                 f"เกณฑ์เริ่มอัด {threshold:.4f}")
    console.line()
    gate.start()
    if mic.gain > 1.05:
        console.note("  (ถ้าเห็นค่าพุ่งชนขอบบ่อย ๆ แปลว่าขยายมากไป ลดด้วย --mic-gain)")

    width, peak, end = 46, 0.0, time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            time.sleep(0.1)
            level = float(live["level"])
            peak = max(peak, level)
            bar = int(min(1.0, level / 0.12) * width)
            hit = int(min(1.0, float(live["threshold"]) / 0.12) * width)
            row = ["·"] * width
            for i in range(bar):
                row[i] = "█"
            if hit < width:
                row[hit] = "┃"
            sys.stdout.write(f"\r  [{''.join(row)}] {level:.4f}  "
                             f"อัดแล้ว {len(heard)} ครั้ง ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        gate.stop()
        mic.stop()

    console.line()
    console.line()
    console.note(f"  ระดับสูงสุดที่วัดได้ {peak:.4f} · เกณฑ์ {threshold:.4f} · "
                 f"จับประโยคได้ {len(heard)} ครั้ง")
    if heard:
        console.line("  ผลรวม: ไมค์ใช้งานได้ ✓")
        return 0
    console.warn("  ผลรวม: ไม่ได้ยินเสียงพูดเลย ✗")
    if peak < threshold:
        console.note("  เสียงเข้ามาเบากว่าเกณฑ์ ลองอย่างใดอย่างหนึ่ง:")
        console.note("    · เพิ่ม input volume:  osascript -e 'set volume input volume 80'")
        console.note(f"    · หรือใส่ MIC_GAIN={max(2.0, threshold / max(peak, 1e-6)):.0f} ใน .env")
    return 1


# ----------------------------------------------------------------- self test
def selftest(cfg: Config) -> int:
    """วาดผลของ `vc.selftest.run_selftest()` ลงเทอร์มินัล — ตรรกะการตรวจอยู่ใน vc/

    ก่อนหน้านี้ที่นี่มีการตรวจอีกชุดหนึ่งที่เขียนซ้ำกับฝั่งเว็บ แล้วพฤติกรรมค่อย ๆ
    ต่างกันไป (P3-21) ตอนนี้ทั้งสองฝ่ายเรียกโค้ดชุดเดียวกัน ต่างกันแค่วิธีแสดงผล
    """
    console = Console()
    result = run_selftest(cfg)
    for i, step in enumerate(result["steps"], 1):
        head = f"[{i}/{len(result['steps'])}] {step['name']}"
        if step["model"]:
            head += f" ({step['model']})"
        console.note(f"{head} ...")
        mark = "✓" if step["ok"] else "✗"
        line = f"      {mark} {step['detail']}"
        if step["ms"]:
            line += f"  ({step['ms'] / 1000:.2f}s)"
        if step["ok"]:
            console.line(line)
        else:
            console.warn(line)
    console.line()
    console.line("ผลรวม: " + ("พร้อมใช้งาน ✓" if result["ok"] else "มีปัญหา ✗"))
    return 0 if result["ok"] else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="คุยกับ AI ด้วยเสียงแบบเรียลไทม์ (ค่าตั้งจาก .env)")
    p.add_argument("--list-devices", action="store_true", help="แสดงอุปกรณ์เสียงทั้งหมด")
    p.add_argument("--selftest", action="store_true", help="ทดสอบ API ครบวงจรโดยไม่ใช้ไมค์")
    p.add_argument("--mic-check", action="store_true",
                   help="มิเตอร์เสียงเข้าแบบสด ดูว่าไมค์ดังพอให้ VAD จับได้ไหม")
    p.add_argument("--input-device", type=int, default=None, help="หมายเลขไมโครโฟน")
    p.add_argument("--output-device", type=int, default=None, help="หมายเลขลำโพง")
    p.add_argument("--model", default=None, help="ทับค่า CHAT_MODEL ใน .env")
    p.add_argument("--headphones", action="store_true",
                   help="ใช้หูฟัง: ปิดระบบกันเสียงลำโพงย้อนเข้าไมค์ (พูดแทรกไวขึ้น)")
    p.add_argument("--no-tts", action="store_true", help="ไม่ต้องออกเสียง แสดง caption อย่างเดียว")
    p.add_argument("--tts-backend", choices=("api", "edge"), default=None,
                   help="ทับค่า TTS_BACKEND: api = TTS_MODEL บนเซิร์ฟเวอร์, "
                        "edge = Microsoft Edge (เลือกเสียงไทยได้ เสียงคงที่ทุกก้อน)")
    p.add_argument("--voice", default=None,
                   help="เสียงของ Edge TTS เช่น th-TH-NiwatNeural (เปิดโหมด edge ให้อัตโนมัติ)")
    p.add_argument("--list-voices", action="store_true",
                   help="แสดงรายชื่อเสียงภาษาไทยของ Microsoft Edge")
    p.add_argument("--no-mic", action="store_true", help="โหมดพิมพ์ ไม่ใช้ไมโครโฟน")
    p.add_argument("--no-web", action="store_true", help="ปิดการค้นข้อมูลจากอินเทอร์เน็ต")
    p.add_argument("--mic-gain", type=float, default=None,
                   help="อัตราขยายเสียงไมค์ (ไม่ใส่ = คำนวณอัตโนมัติ)")
    p.add_argument("--one-voice", action="store_true",
                   help="รอคำตอบจบแล้วค่อยพูดทีเดียว เสียงจะเป็นคนเดียวกันแต่เริ่มพูดช้าลง")
    p.add_argument("--stream-tts", action="store_true",
                   help="(ค่าเริ่มต้นอยู่แล้ว) พูดไปพร้อมกับที่ข้อความกำลังไหล")
    p.add_argument("--no-greet", dest="greet", action="store_false", help="ไม่ต้องทักทายตอนเริ่ม")
    p.add_argument("--save-audio", action="store_true", help="เก็บไฟล์เสียงที่ผู้ใช้พูดไว้ด้วย")
    p.add_argument("--threshold", type=float, default=None, help="ทับค่า VAD_ABS_THRESHOLD")
    p.set_defaults(greet=True)
    args = p.parse_args()

    if args.list_devices:
        print(list_devices())
        return 0

    if args.list_voices:
        from vc.tts_edge import THAI_VOICES
        print("เสียงภาษาไทยของ Microsoft Edge (ใช้กับ --voice หรือ EDGE_TTS_VOICE):")
        for name, label, _gender in THAI_VOICES:
            print(f"  {name:<26} {label}")
        return 0

    cfg = load_config()
    if args.model:
        cfg.chat_model = args.model
    if args.no_tts:
        cfg.tts_enabled = False
    if args.no_web:
        cfg.web_search = False
    if args.headphones:
        cfg.echo_guard = False
    if args.save_audio:
        cfg.save_audio = True
    if args.threshold is not None:
        cfg.vad_abs_threshold = args.threshold
    if args.mic_gain is not None:
        cfg.mic_gain = args.mic_gain
    if args.stream_tts:
        cfg.tts_single_request = False
    if args.one_voice:
        cfg.tts_single_request = True
    if args.tts_backend:
        cfg.tts_backend = args.tts_backend
    if args.voice:
        # ระบุเสียงมาแต่ยังอยู่แบ็กเอนด์ api จะไม่มีผลอะไรเลย — สลับให้เลย
        cfg.edge_voice = args.voice
        cfg.tts_backend = "edge"
    cfg.input_device = args.input_device
    cfg.output_device = args.output_device

    if args.selftest:
        return selftest(cfg)

    if args.mic_check:
        return mic_check(cfg)

    if not cfg.asr_model and not args.no_mic:
        print("ไม่พบ ASR_MODEL ใน .env — ใช้ --no-mic เพื่อคุยแบบพิมพ์", file=sys.stderr)
        return 2

    # แปลง argument ของ CLI เป็นสัญญาที่แกนกลางประกาศไว้เอง — แกนกลางไม่รู้จัก argparse
    chat = VoiceChat(cfg, RuntimeOptions(no_mic=args.no_mic, greet=args.greet))
    chat.run()
    if chat.audio_stuck:
        # CoreAudio ค้างอยู่ ปล่อยให้ Python ปิดตัวตามปกติจะแขวนใน Py_Finalize
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
