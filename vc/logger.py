"""บันทึกการสนทนา: JSONL (เหตุการณ์ + latency) และ Markdown (อ่านง่าย)

สิ่งที่เขียนลงดิสก์คือคำพูดจริงของผู้ใช้ทั้งหมด จึงถือเป็นข้อมูลอ่อนไหว (P3-18)
1. โฟลเดอร์ที่สร้างใหม่เป็น 0700 และไฟล์เป็น 0600 — เจ้าของเครื่องอ่านได้คนเดียว
2. `LOG_TRANSCRIPT=0` ปิดการเขียนตัวข้อความได้ทั้งหมด (ยังเก็บ latency ไว้วัดผล)
3. `LOG_RETENTION_DAYS=N` ลบ session ที่เก่ากว่า N วันตอนเริ่มโปรแกรม (0 = ไม่ลบ)
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .api import pcm16_to_wav

DIR_MODE = 0o700         # drwx------
FILE_MODE = 0o600        # -rw-------
SESSION_PREFIX = "session-"
STAMP_FORMAT = "%Y%m%d-%H%M%S"

# ฟิลด์ที่มีคำพูดจริงอยู่ข้างใน — ถูกตัดออกเมื่อปิดการเก็บ transcript
TEXT_FIELDS = ("text", "spoken", "arg", "detail", "query")


def chmod_quiet(path: Path, mode: int) -> None:
    """ตั้งสิทธิ์แบบไม่ทำให้โปรแกรมล้ม (บางระบบไฟล์/Windows ทำไม่ได้)"""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def private_dir(path: Path) -> Path:
    """สร้างโฟลเดอร์ให้เจ้าของอ่านได้คนเดียว — คืน path เดิมเพื่อเรียกต่อกันได้

    ต้อง chmod หลัง mkdir เพราะ `mode=` ของ mkdir ถูก umask ของผู้ใช้หักออก
    (umask 022 ปกติจะได้ 0755 ทั้งที่ขอ 0700 ไป)
    """
    path.mkdir(parents=True, exist_ok=True)
    chmod_quiet(path, DIR_MODE)
    return path


def private_file(path: Path) -> Path:
    """สร้างไฟล์เปล่าด้วยสิทธิ์ 0600 ตั้งแต่ตอนเกิด (ไม่มีจังหวะที่ไฟล์เปิดกว้าง)"""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, FILE_MODE)
    os.close(fd)
    chmod_quiet(path, FILE_MODE)
    return path


def session_started_at(path: Path) -> datetime | None:
    """อ่านเวลาเริ่มจากชื่อโฟลเดอร์ `session-YYYYmmdd-HHMMSS`"""
    if not path.name.startswith(SESSION_PREFIX):
        return None
    try:
        return datetime.strptime(path.name[len(SESSION_PREFIX):], STAMP_FORMAT)
    except ValueError:
        return None


def purge_old_sessions(root: Path, days: int,
                       now: datetime | None = None) -> list[Path]:
    """ลบโฟลเดอร์ session ที่เก่ากว่า `days` วัน — คืนรายการที่ลบไปจริง

    `days <= 0` = ไม่ลบอะไรเลย (ค่าเริ่มต้น) เพราะบันทึกเป็นข้อมูลของผู้ใช้
    การลบทิ้งเองโดยไม่ได้สั่งถือว่าอันตรายกว่าการเก็บไว้
    อ่านอายุจากชื่อโฟลเดอร์ ไม่ใช่ mtime เพราะการคัดลอก/ย้ายไฟล์รีเซ็ต mtime ได้
    """
    if days <= 0 or not root.is_dir():
        return []
    cutoff = (now or datetime.now()) - timedelta(days=days)
    removed: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        started = session_started_at(child)
        if started is None or started >= cutoff:
            continue
        try:
            shutil.rmtree(child)
        except OSError:
            continue
        removed.append(child)
    return removed


class SessionLogger:
    def __init__(self, root: Path, save_audio: bool = False,
                 meta: dict | None = None, *, transcript: bool = True,
                 retention_days: int = 0):
        self.started = datetime.now()
        self.transcript = transcript
        root = Path(root)
        private_dir(root)
        self.purged = purge_old_sessions(root, retention_days)
        stamp = self.started.strftime(STAMP_FORMAT)
        self.dir = private_dir(root / f"{SESSION_PREFIX}{stamp}")
        self.jsonl = private_file(self.dir / "session.jsonl")
        self.md = self.dir / "transcript.md"
        self.audio_dir = self.dir / "audio"
        # ไม่มี transcript ก็ไม่มีเหตุผลให้เก็บไฟล์เสียงของผู้ใช้ด้วย
        self.save_audio = save_audio and transcript
        if self.save_audio:
            private_dir(self.audio_dir)
        self._lock = threading.Lock()
        self._turns = 0

        if self.transcript:
            private_file(self.md)
            self.md.write_text(
                f"# บันทึกการสนทนาด้วยเสียง\n\n"
                f"- เริ่ม: {self.started.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "".join(f"- {k}: `{v}`\n" for k, v in (meta or {}).items())
                + "\n---\n\n",
                encoding="utf-8",
            )
        if self.purged:
            self.event("logs_purged", count=len(self.purged),
                       days=retention_days)
        self.event("session_start", meta=meta or {},
                   transcript=self.transcript)

    # ---------------------------------------------------------------- events
    def _scrub(self, fields: dict) -> dict:
        """ตัดคำพูดออกจากเหตุการณ์เมื่อปิดการเก็บ transcript

        ยังเก็บ latency/ขนาด/ชนิดไว้ทั้งหมด เพราะเป็นตัวเลขที่ไม่ระบุตัวบุคคล
        และเป็นข้อมูลเดียวที่ใช้ตามหาว่าช้าตรงไหนตอนมีปัญหา
        """
        if self.transcript:
            return fields
        out = {}
        for key, value in fields.items():
            if key in TEXT_FIELDS and isinstance(value, str):
                out[f"{key}_chars"] = len(value)
                continue
            if key == "sources" and value:
                out["sources"] = len(value)
                continue
            out[key] = value
        return out

    def event(self, kind: str, **fields) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"), "type": kind}
        rec.update(self._scrub(fields))
        with self._lock, self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def turn(self, role: str, text: str, **fields) -> None:
        self._turns += 1
        self.event("message", role=role, text=text, **fields)
        if not self.transcript:
            return
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
        private_file(path)
        path.write_bytes(pcm16_to_wav(pcm, samplerate))
        chmod_quiet(path, FILE_MODE)
        return str(path.relative_to(self.dir))

    def close(self, reason: str = "normal") -> None:
        dur = time.time() - self.started.timestamp()
        self.event("session_end", reason=reason, turns=self._turns, seconds=round(dur, 1))
        if not self.transcript:
            return
        with self._lock, self.md.open("a", encoding="utf-8") as f:
            f.write(
                f"\n---\n\nจบการสนทนา: {datetime.now().strftime('%H:%M:%S')} "
                f"(รวม {int(dur // 60)} นาที {int(dur % 60)} วินาที, {self._turns} ข้อความ)\n"
            )
