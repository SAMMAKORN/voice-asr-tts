"""ไคลเอนต์เรียก LiteLLM: chat (streaming), ASR, TTS"""
from __future__ import annotations

import io
import json
import threading
import wave
from typing import Callable, Iterator

import httpx
import numpy as np

from .config import Config


class ApiError(RuntimeError):
    pass


def _merge_tool_deltas(acc: dict[int, dict], deltas: list[dict]) -> None:
    """ประกอบ tool_call ที่สตรีมมาเป็นชิ้น ๆ (ชื่อมาก่อน อาร์กิวเมนต์ทยอยตามมา)"""
    for d in deltas:
        idx = d.get("index")
        if idx is None:
            idx = max(acc) if acc else 0
        slot = acc.setdefault(int(idx), {"id": "", "name": "", "args": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]


def pcm16_to_wav(pcm: np.ndarray, samplerate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def wav_to_pcm16(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ApiError(f"รองรับเฉพาะ WAV 16-bit (ได้ {width * 8}-bit)")
    pcm = np.frombuffer(raw, dtype="<i2")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return pcm.copy(), sr


def resample_i16(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or pcm.size == 0:
        return pcm
    n_out = int(round(pcm.size * dst_sr / src_sr))
    x = np.linspace(0.0, 1.0, pcm.size, endpoint=False)
    xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xi, x, pcm.astype(np.float32)).astype(np.int16)


class ApiClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Accept": "application/json",
        }
        # timeout ยาวสำหรับ read เพราะ TTS/ASR ใช้เวลาสังเคราะห์
        self._chat = httpx.Client(
            base_url=cfg.base_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        self._audio = httpx.Client(
            base_url=cfg.base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    def close(self) -> None:
        self._chat.close()
        self._audio.close()

    # ------------------------------------------------------------------ chat
    def _stream_once(
        self, messages: list[dict], cancel: threading.Event,
        tools: list[dict] | None, tool_choice: str = "auto",
    ) -> Iterator[tuple[str, object]]:
        """สตรีมหนึ่งรอบ คืนเป็นคู่ ("text", ข้อความ) หรือ ("tool", ชิ้นส่วน tool_call)"""
        payload: dict = {
            "model": self.cfg.chat_model,
            "messages": messages,
            "stream": True,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        with self._chat.stream("POST", "/v1/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                r.read()
                raise ApiError(f"chat {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if cancel.is_set():
                    return
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    if chunk == "[DONE]":
                        return
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield "text", text
                calls = delta.get("tool_calls")
                if calls:
                    yield "tool", calls

    def chat_stream(
        self, messages: list[dict], cancel: threading.Event,
        tools: list[dict] | None = None,
        run_tool: Callable[[str, dict], str] | None = None,
        max_rounds: int = 2,
    ) -> Iterator[str]:
        """สตรีมคำตอบทีละ token; หยุดทันทีเมื่อ cancel ถูกตั้ง

        ถ้าส่ง tools + run_tool มาด้วย และโมเดลขอเรียกเครื่องมือ (เช่น ค้นเว็บ)
        จะรันให้แล้วส่งผลกลับเข้าไปให้โมเดลตอบต่อ วนได้สูงสุด max_rounds รอบ
        ข้อความที่ yield ออกไปยังเป็น text ล้วนเหมือนเดิม ฝั่ง TTS จึงไม่ต้องแก้อะไร
        """
        msgs = list(messages)
        can_use = bool(tools and run_tool)
        spoke = False
        used_tools = False

        for rnd in range(max_rounds + 1):
            if cancel.is_set():
                return          # ถูกพูดแทรกตอนกำลังค้นข้อมูล — ไม่ต้องยิงต่อให้เปลืองแล้ว

            # รอบสุดท้ายยังต้องแนบ tools ไปด้วย ไม่งั้นโมเดลอ่านประวัติ tool_calls
            # ที่อยู่ในบทสนทนาไม่ออก แล้วตอบกลับมาเป็นค่าว่าง (เจอจริงเป็นครั้งคราว)
            # จึงใช้ tool_choice=none ห้ามเรียกเพิ่มแทนการตัด tools ทิ้ง
            choice = "auto" if rnd < max_rounds else "none"
            calls: dict[int, dict] = {}
            said = ""
            for kind, payload in self._stream_once(
                    msgs, cancel, tools if can_use else None, choice):
                if kind == "text":
                    said += payload  # type: ignore[operator]
                    spoke = True
                    yield payload  # type: ignore[misc]
                else:
                    _merge_tool_deltas(calls, payload)  # type: ignore[arg-type]

            wanted = [calls[k] for k in sorted(calls) if calls[k]["name"]]
            if cancel.is_set():
                return
            if not wanted or run_tool is None:
                break
            used_tools = True

            assistant: dict = {
                "role": "assistant",
                "tool_calls": [
                    {"id": c["id"] or f"call_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                    for i, c in enumerate(wanted)
                ],
            }
            if said.strip():
                assistant["content"] = said
            msgs.append(assistant)

            for i, c in enumerate(wanted):
                if cancel.is_set():
                    return
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({
                    "role": "tool",
                    "tool_call_id": c["id"] or f"call_{i}",
                    "content": run_tool(c["name"], args if isinstance(args, dict) else {}),
                })

        # ค้นข้อมูลมาแล้วแต่ไม่ยอมพูดอะไรเลย — กระทุ้งอีกครั้ง ดีกว่าปล่อยเงียบใส่ผู้ใช้
        if used_tools and not spoke and not cancel.is_set():
            msgs.append({"role": "user",
                         "content": "ตอบคำถามเดิมสั้น ๆ จากข้อมูลที่ค้นมาได้เลย"})
            for kind, payload in self._stream_once(msgs, cancel, None):
                if kind == "text":
                    yield payload  # type: ignore[misc]

    # ------------------------------------------------------------------- asr
    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        wav = pcm16_to_wav(pcm, samplerate)
        files = {"file": ("speech.wav", wav, "audio/wav")}
        data = {"model": self.cfg.asr_model}
        r = self._audio.post("/v1/audio/transcriptions", files=files, data=data)
        if r.status_code >= 400:
            raise ApiError(f"asr {r.status_code}: {r.text[:300]}")
        try:
            return (r.json().get("text") or "").strip()
        except json.JSONDecodeError:
            raise ApiError(f"asr ตอบกลับผิดรูปแบบ: {r.text[:200]}")

    # ------------------------------------------------------------------- tts
    def synthesize(self, text: str, dst_sr: int) -> np.ndarray:
        """OmniVoice ผ่าน LiteLLM รับได้แค่ {model, input} — ใส่ voice/format แล้ว 500

        ถ้าตั้ง TTS_BACKEND=edge จะไปเรียก Microsoft Edge แทน (เลือกเสียงได้
        และเสียงคงที่ทุกก้อน) — import ตรงนี้เพื่อไม่ให้ import วนกันกับ tts_edge
        """
        if self.cfg.use_edge_tts:
            from .tts_edge import synthesize as edge_synthesize

            return edge_synthesize(
                text, dst_sr,
                voice=self.cfg.edge_voice,
                rate=self.cfg.edge_rate,
                volume=self.cfg.edge_volume,
                pitch=self.cfg.edge_pitch,
            )

        payload = {"model": self.cfg.tts_model, "input": text}
        r = self._audio.post("/v1/audio/speech", json=payload)
        if r.status_code >= 400:
            raise ApiError(f"tts {r.status_code}: {r.text[:300]}")
        body = r.content
        if body[:4] != b"RIFF":
            raise ApiError(f"tts ไม่ได้คืนเสียง: {body[:200]!r}")
        pcm, sr = wav_to_pcm16(body)
        return resample_i16(pcm, sr, dst_sr)
