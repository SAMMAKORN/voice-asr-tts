"""ไคลเอนต์เรียก LiteLLM: chat (streaming), ASR, TTS

**นี่คือขอบเขตจัดการข้อผิดพลาดเพียงจุดเดียวของฝั่ง HTTP (P2-6)** — ทุกความผิดพลาด
ที่ออกจากไฟล์นี้ต้องเป็น `ApiError` เท่านั้น ห้ามให้ `httpx.*` หลุดขึ้นไปให้ชั้นบน
ดักเอง (บั๊กเดิม: `httpx.ConnectError` ตอนถอดเสียงทะลุถึง `run()` แล้วปิดทั้ง session)
พร้อมลองใหม่แบบ exponential backoff + jitter สำหรับความผิดพลาดที่ลองใหม่แล้วมีผล
"""
from __future__ import annotations

import io
import json
import logging
import random
import re
import threading
import time
import wave
from typing import Callable, Iterator

import httpx
import numpy as np

from .config import Config

log = logging.getLogger("voicechat.api")

# ความผิดพลาดชั่วคราวระดับเครือข่าย — ลองใหม่ได้ (NetworkError ครอบ Connect/Read/
# Write/CloseError, TimeoutException ครอบ connect/read/write/pool timeout)
RETRYABLE_EXC = (httpx.TimeoutException, httpx.NetworkError,
                 httpx.RemoteProtocolError)
BACKOFF_CAP = 8.0          # เพดานเวลาคอยต่อครั้ง (วินาที)


class ApiError(RuntimeError):
    pass


# สถานะ HTTP -> สาเหตุแบบที่บอกผู้ใช้ได้ว่า "ต้องไปแก้ตรงไหน"
_WHY_STATUS = {
    401: "API key ไม่ถูกต้องหรือหมดอายุ (401) — ตรวจ API_KEY ใน .env",
    403: "บัญชีนี้ไม่มีสิทธิ์เรียกโมเดลที่ตั้งไว้ (403) — ตรวจชื่อโมเดลใน .env",
    404: "ไม่พบโมเดลหรือปลายทางนี้ (404) — ตรวจชื่อโมเดลและ API_BASE_URL ใน .env",
    413: "ข้อความยาวเกินที่เซิร์ฟเวอร์รับไหว (413)",
    429: "ถูกจำกัดจำนวนครั้งที่เรียก (429) — รอสักครู่แล้วลองใหม่",
}
_STATUS = re.compile(r"^\w+ (\d{3}):")


def explain(error: str) -> str:
    """แปลข้อความผิดพลาดดิบให้เป็นประโยคเดียวที่ผู้ใช้อ่านรู้เรื่อง

    ตัวเต็ม (JSON ของเซิร์ฟเวอร์, repr ของ exception) ยังถูกเก็บครบใน
    session.jsonl และแผงบันทึกระบบเสมอ ที่นี่คือสิ่งที่ไปโผล่ในช่องแชท จึงต้อง
    สั้นและบอกทางแก้ — ผู้ใช้ที่เห็น "chat 403: {\"error\":{...}}" กลางบทสนทนา
    ได้ข้อมูลเท่ากับไม่เห็นอะไรเลย
    """
    m = _STATUS.match(error)
    if m:
        status = int(m.group(1))
        if status in _WHY_STATUS:
            return _WHY_STATUS[status]
        if status >= 500:
            return f"เซิร์ฟเวอร์โมเดลขัดข้อง ({status}) — ลองใหม่อีกครั้ง"
        return f"เซิร์ฟเวอร์ตอบกลับเป็นข้อผิดพลาด ({status})"
    if "เชื่อมต่อไม่สำเร็จ" in error:
        return "ต่อเซิร์ฟเวอร์ไม่ได้ — ตรวจ API_BASE_URL ใน .env และการเชื่อมต่อเน็ต"
    if "สตรีมขาดกลางทาง" in error:
        return "การเชื่อมต่อขาดกลางคัน — ลองถามใหม่อีกครั้ง"
    if "Timeout" in error or "timeout" in error:
        return "เซิร์ฟเวอร์ไม่ตอบภายในเวลาที่กำหนด — ลองใหม่อีกครั้ง"
    return "เรียกโมเดลไม่สำเร็จ — ดูรายละเอียดในแผงบันทึกระบบ"


def _retryable_status(status: int) -> bool:
    """429 (โดนจำกัดอัตรา) และ 5xx เท่านั้น — 4xx อื่นลองใหม่ไปก็ผิดเหมือนเดิม"""
    return status == 429 or status >= 500


def _retry_after(response: httpx.Response) -> float | None:
    """อ่าน `Retry-After` (วินาที) ถ้าเซิร์ฟเวอร์บอกมา"""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None          # รูปแบบวันที่ (HTTP-date) — ใช้ backoff ปกติแทน


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
        # สร้างไว้ตั้งแต่ต้นแม้ยังไม่โคลน (ตัวมันอ่านไฟล์แบบ lazy) — สร้างทีหลัง
        # ในเธรด TTS จะซ้อนกันได้เมื่อสองก้อนแรกออกพร้อมกัน
        self._voice_ref = None
        if cfg.clone_enabled:
            from .voiceclone import VoiceReference

            self._voice_ref = VoiceReference(
                cfg.tts_ref_audio,
                text=cfg.tts_ref_text,
                max_sec=cfg.tts_ref_max_sec,
                normalize=cfg.tts_ref_normalize,
            )
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

    # ------------------------------------------------ ลองใหม่ + แปลงข้อผิดพลาด
    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """เวลาคอยก่อนลองครั้งถัดไป — เคารพ Retry-After ถ้ามี ไม่งั้นเลขคู่ทวี + jitter"""
        if retry_after is not None:
            return min(retry_after, BACKOFF_CAP)
        base = max(0, self.cfg.http_retry_base_ms) / 1000.0
        # jitter ±20% กันหลาย client ยิงกลับมาพร้อมกันเป็นระลอก
        return min(BACKOFF_CAP, base * (2 ** attempt) * random.uniform(0.8, 1.2))

    def _can_retry(self, attempt: int) -> bool:
        return attempt < max(0, self.cfg.http_retry_max)

    def _wait(self, label: str, attempt: int, delay: float, reason: str,
              cancel: threading.Event | None = None) -> bool:
        """คอยตาม backoff — คืน False ถ้าถูกสั่งยกเลิกระหว่างคอย"""
        log.warning("%s ล้มเหลว (%s) — ลองใหม่ครั้งที่ %d ในอีก %.2f วินาที",
                    label, reason, attempt + 1, delay)
        if cancel is not None:
            return not cancel.wait(delay)
        time.sleep(delay)
        return True

    def _post(self, client: httpx.Client, url: str, label: str, **kw) -> httpx.Response:
        """POST ที่ลองใหม่ให้เองและคืนได้เฉพาะผลสำเร็จ (ไม่งั้นโยน ApiError)"""
        attempt = 0
        while True:
            try:
                r = client.post(url, **kw)
            except RETRYABLE_EXC as exc:
                if not self._can_retry(attempt):
                    raise ApiError(f"{label} เชื่อมต่อไม่สำเร็จ: {exc!r}") from exc
                self._wait(label, attempt, self._delay(attempt, None), repr(exc))
                attempt += 1
                continue
            except httpx.HTTPError as exc:      # ผิดตั้งแต่รูปแบบคำขอ ลองใหม่ไม่ช่วย
                raise ApiError(f"{label} เรียกไม่สำเร็จ: {exc!r}") from exc
            if r.status_code >= 400:
                if _retryable_status(r.status_code) and self._can_retry(attempt):
                    self._wait(label, attempt,
                               self._delay(attempt, _retry_after(r)),
                               f"HTTP {r.status_code}")
                    attempt += 1
                    continue
                raise ApiError(f"{label} {r.status_code}: {r.text[:300]}")
            return r

    # ------------------------------------------------------------------ chat
    def _stream_once(
        self, messages: list[dict], cancel: threading.Event,
        tools: list[dict] | None, tool_choice: str = "auto",
        allow_retry: bool = True, temperature: float | None = None,
    ) -> Iterator[tuple[str, object]]:
        """สตรีมหนึ่งรอบ คืนเป็นคู่ ("text", ข้อความ) หรือ ("tool", ชิ้นส่วน tool_call)

        ลองใหม่ได้เฉพาะตอนที่ **ยังไม่มี token ไหลออกไปเลย** — ถ้าเริ่มพูดแล้วค่อยหลุด
        การยิงซ้ำจะทำให้ผู้ใช้ได้ยินท่อนเดิมสองครั้ง (AC-6.4)
        """
        payload: dict = {
            "model": self.cfg.chat_model,
            "messages": messages,
            "stream": True,
            "temperature": (self.cfg.temperature if temperature is None
                            else temperature),
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        attempt = 0
        while True:
            emitted = False
            retry: tuple[float, str] | None = None
            try:
                with self._chat.stream("POST", "/v1/chat/completions",
                                       json=payload) as r:
                    if r.status_code >= 400:
                        r.read()
                        if (_retryable_status(r.status_code) and allow_retry
                                and self._can_retry(attempt)):
                            retry = (self._delay(attempt, _retry_after(r)),
                                     f"HTTP {r.status_code}")
                        else:
                            raise ApiError(f"chat {r.status_code}: {r.text[:300]}")
                    else:
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
                                emitted = True
                                yield "text", text
                            calls = delta.get("tool_calls")
                            if calls:
                                emitted = True
                                yield "tool", calls
                        return
            except ApiError:
                raise
            except RETRYABLE_EXC as exc:
                if allow_retry and not emitted and self._can_retry(attempt):
                    retry = (self._delay(attempt, None), repr(exc))
                else:
                    raise ApiError(f"chat สตรีมขาดกลางทาง: {exc!r}") from exc
            except httpx.HTTPError as exc:
                raise ApiError(f"chat เรียกไม่สำเร็จ: {exc!r}") from exc

            if retry is None:
                return
            delay, reason = retry
            if not self._wait("chat", attempt, delay, reason, cancel):
                return                     # ถูกพูดแทรกระหว่างคอย — เลิกยิงต่อ
            attempt += 1

    def chat_stream(
        self, messages: list[dict], cancel: threading.Event,
        tools: list[dict] | None = None,
        run_tool: Callable[[str, dict], str] | None = None,
        max_rounds: int = 2, temperature: float | None = None,
    ) -> Iterator[str]:
        """สตรีมคำตอบทีละ token; หยุดทันทีเมื่อ cancel ถูกตั้ง

        ถ้าส่ง tools + run_tool มาด้วย และโมเดลขอเรียกเครื่องมือ (เช่น ค้นเว็บ)
        จะรันให้แล้วส่งผลกลับเข้าไปให้โมเดลตอบต่อ วนได้สูงสุด max_rounds รอบ
        ข้อความที่ yield ออกไปยังเป็น text ล้วนเหมือนเดิม ฝั่ง TTS จึงไม่ต้องแก้อะไร

        `temperature` ทับค่าใน .env เฉพาะการเรียกครั้งนี้ — คำทักทาย (vc/greeting.py)
        ต้องการความหลากหลายสูงกว่าการสนทนาปกติ ส่วนบทสนทนาที่ผู้ใช้จูนไว้เองต้องไม่
        ถูกกระทบ · None = ใช้ค่าจาก .env ตามเดิม
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
            # พูดออกไปแล้วห้ามยิงซ้ำ ไม่งั้นผู้ใช้ได้ยินท่อนเดิมสองครั้ง (AC-6.4)
            for kind, payload in self._stream_once(
                    msgs, cancel, tools if can_use else None, choice,
                    allow_retry=not spoke, temperature=temperature):
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
            for kind, payload in self._stream_once(msgs, cancel, None,
                                                   allow_retry=False):
                if kind == "text":
                    yield payload  # type: ignore[misc]

    # ------------------------------------------------------------------- asr
    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        wav = pcm16_to_wav(pcm, samplerate)
        files = {"file": ("speech.wav", wav, "audio/wav")}
        data = {"model": self.cfg.asr_model}
        r = self._post(self._audio, "/v1/audio/transcriptions", "asr",
                       files=files, data=data)
        try:
            return (r.json().get("text") or "").strip()
        except (json.JSONDecodeError, AttributeError, ValueError):
            raise ApiError(f"asr ตอบกลับผิดรูปแบบ: {r.text[:200]}") from None

    # ------------------------------------------------------------------- tts
    def _reference_payload(self) -> dict | None:
        """เสียงอ้างอิงสำหรับโคลน โหลดครั้งเดียวแล้วจำไว้ (None = ไม่โคลน)

        ส่ง `transcribe` เข้าไปด้วยเพื่อให้ถอดคำอ่านเองได้เมื่อผู้ใช้เอาไฟล์เสียง
        ตัวเองมาวางโดยไม่มี .txt คู่กัน — เรียกครั้งเดียวตอนก้อนแรกเท่านั้น
        """
        if not self.cfg.clone_enabled or self._voice_ref is None:
            return None
        return self._voice_ref.payload(self.transcribe)

    def synthesize(self, text: str, dst_sr: int) -> np.ndarray:
        """OmniVoice ผ่าน LiteLLM รับ {model, input} + คู่ {ref_audio, ref_text}

        `voice`/`speed`/`response_format` ยังใส่ไม่ได้เหมือนเดิม (500 หรือถูกเมิน)
        แต่ **โคลนเสียงได้** เมื่อส่ง ref_audio เป็น data URI พร้อม ref_text ที่ตรงกัน
        ซึ่งทำให้เสียงคงที่ทุกก้อนแทนที่จะสุ่มใหม่ทุก request — ดู vc/voiceclone.py

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
        ref = self._reference_payload()
        if ref:
            payload.update(ref)
        r = self._post(self._audio, "/v1/audio/speech", "tts", json=payload)
        body = r.content
        if body[:4] != b"RIFF":
            raise ApiError(f"tts ไม่ได้คืนเสียง: {body[:200]!r}")
        try:
            pcm, sr = wav_to_pcm16(body)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — WAV เสียหาย ต้องไม่หลุดเป็นชนิดอื่น
            raise ApiError(f"tts อ่านไฟล์เสียงไม่ได้: {exc!r}") from exc
        return resample_i16(pcm, sr, dst_sr)
