"""บันทึกการสนทนา: JSONL (เหตุการณ์ + latency) และ Markdown (อ่านง่าย)"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .api import pcm16_to_wav


class SessionLogger:
    def __init__(self, root: Path, save_audio: bool = False, meta: dict | None = None):
        self.started = datetime.now()
        stamp = self.started.strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"session-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "session.jsonl"
        self.md = self.dir / "transcript.md"
        self.audio_dir = self.dir / "audio"
        self.save_audio = save_audio
        if save_audio:
            self.audio_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._turns = 0

        self.md.write_text(
            f"# บันทึกการสนทนาด้วยเสียง\n\n"
            f"- เริ่ม: {self.started.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "".join(f"- {k}: `{v}`\n" for k, v in (meta or {}).items())
            + "\n---\n\n",
            encoding="utf-8",
        )
        self.event("session_start", meta=meta or {})

    # ---------------------------------------------------------------- events
    def event(self, kind: str, **fields) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"), "type": kind}
        rec.update(fields)
        with self._lock, self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def turn(self, role: str, text: str, **fields) -> None:
        self._turns += 1
        self.event("message", role=role, text=text, **fields)
        who = {"user": "🧑 คุณ", "assistant": "🤖 AI", "system": "⚙️ ระบบ"}.get(role, role)
        extra = ""
        if fields.get("interrupted"):
            extra = "  \n> _(ถูกพูดขัดกลางประโยค)_"
        for i, s in enumerate(fields.get("sources") or [], 1):
            extra += f"\n> 🔎 [{i}] [{s.get('title') or s.get('url')}]({s.get('url')})"
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock, self.md.open("a", encoding="utf-8") as f:
            f.write(f"**{who}** `{stamp}`\n\n{text.strip()}{extra}\n\n")

    def save_utterance(self, pcm: np.ndarray, samplerate: int, index: int) -> str | None:
        if not self.save_audio:
            return None
        path = self.audio_dir / f"user-{index:03d}.wav"
        path.write_bytes(pcm16_to_wav(pcm, samplerate))
        return str(path.relative_to(self.dir))

    def close(self, reason: str = "normal") -> None:
        dur = time.time() - self.started.timestamp()
        self.event("session_end", reason=reason, turns=self._turns, seconds=round(dur, 1))
        with self._lock, self.md.open("a", encoding="utf-8") as f:
            f.write(
                f"\n---\n\nจบการสนทนา: {datetime.now().strftime('%H:%M:%S')} "
                f"(รวม {int(dur // 60)} นาที {int(dur % 60)} วินาที, {self._turns} ข้อความ)\n"
            )
