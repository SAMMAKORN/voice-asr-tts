"""แสดง caption บทสนทนาบนเทอร์มินัลแบบเรียลไทม์"""
from __future__ import annotations

import shutil
import sys
import threading
import unicodedata
from urllib.parse import urlparse

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"
MAGENTA = "\033[35m"


def _width(text: str) -> int:
    return sum(0 if unicodedata.combining(c) else 1 for c in text)


class Console:
    """เขียน caption ทีละ token พร้อมตัดบรรทัดและย่อหน้าต่อเนื่อง"""

    def __init__(self, use_color: bool = True):
        self.color = use_color and sys.stdout.isatty()
        self._lock = threading.RLock()
        self._col = 0
        self._indent = 0
        self._open = False
        self._status = ""

    # ------------------------------------------------------------ พื้นฐาน
    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    @property
    def width(self) -> int:
        return max(40, shutil.get_terminal_size((100, 24)).columns - 1)

    def _clear_status(self) -> None:
        if self._status:
            sys.stdout.write("\r\033[K" if self.color else "\n")
            self._status = ""

    # ------------------------------------------------------------ สถานะ
    def status(self, icon: str, text: str, color: str = GRAY) -> None:
        with self._lock:
            line = f"{icon} {text}"
            if self._open or line == self._status:
                return
            self._clear_status()
            sys.stdout.write("\r\033[K" + self._c(color, line) if self.color
                             else line + "\n")
            sys.stdout.flush()
            self._status = line

    def clear_status(self) -> None:
        with self._lock:
            self._clear_status()
            sys.stdout.flush()

    # ------------------------------------------------------------ caption
    def begin(self, label: str, color: str, role: str = "") -> None:
        """role ไม่ใช้บนเทอร์มินัล แต่ WebConsole ใช้แยกว่าใครเป็นคนพูด"""
        with self._lock:
            self._clear_status()
            if self._open:
                self.end()
            prefix = f"{label} "
            sys.stdout.write(self._c(color + BOLD, prefix) + self._c(GRAY, "▏"))
            sys.stdout.flush()
            self._indent = _width(prefix) + 1
            self._col = self._indent
            self._open = True

    def write(self, text: str) -> None:
        with self._lock:
            if not self._open:
                return
            limit = self.width
            out: list[str] = []
            for ch in text:
                if ch == "\n":
                    out.append("\n" + " " * self._indent)
                    self._col = self._indent
                    continue
                if self._col >= limit:
                    out.append("\n" + " " * self._indent)
                    self._col = self._indent
                    if ch == " ":
                        continue
                out.append(ch)
                self._col += 0 if unicodedata.combining(ch) else 1
            sys.stdout.write("".join(out))
            sys.stdout.flush()

    def end(self, suffix: str = "") -> None:
        with self._lock:
            if not self._open:
                return
            if suffix:
                sys.stdout.write(self._c(YELLOW, suffix))
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._open = False
            self._col = 0

    # ------------------------------------------------------------ ข้อความอื่น
    def line(self, text: str = "") -> None:
        with self._lock:
            self._clear_status()
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def note(self, text: str) -> None:
        self.line(self._c(GRAY, text))

    def warn(self, text: str) -> None:
        self.line(self._c(YELLOW, "⚠ " + text))

    def error(self, text: str) -> None:
        self.line(self._c(RED, "✗ " + text))

    def sources(self, items: list[dict]) -> None:
        """แสดงแหล่งที่มาของข้อมูลที่ค้นมาจากอินเทอร์เน็ต"""
        if not items:
            return
        self.line(self._c(GRAY, "  🔎 แหล่งที่มา"))
        for i, s in enumerate(items[:6], 1):
            host = s.get("host") or urlparse(s.get("url", "")).netloc.removeprefix("www.")
            title = (s.get("title") or "").strip()[:64]
            self.line(self._c(GRAY, f"     {i}. {host}") + self._c(DIM, f"  {title}"))
