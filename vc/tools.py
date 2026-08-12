"""เครื่องมือที่ให้ LLM เรียกใช้เพื่อหาข้อมูลปัจจุบันจากอินเทอร์เน็ต"""
from __future__ import annotations

from . import websearch
from .config import Config

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "ค้นข้อมูลจากอินเทอร์เน็ต ใช้เมื่อผู้ใช้ถามถึงเรื่องที่เปลี่ยนตามเวลา "
                "หรือเรื่องที่เกิดหลังจากข้อมูลที่คุณมี เช่น ราคา สภาพอากาศ ข่าว "
                "ผลกีฬา อัตราแลกเปลี่ยน ตารางเวลา หรือข้อเท็จจริงที่ต้องอ้างอิงแหล่งที่มา"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "คำค้นสั้น ๆ ตรงประเด็น ใช้ภาษาเดียวกับที่ผู้ใช้ถาม",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": (
                "เปิดอ่านเนื้อหาหน้าเว็บจากลิงก์ที่ได้จาก web_search "
                "ใช้เฉพาะตอนที่คำอธิบายย่อจากผลค้นหายังไม่พอตอบ เพราะใช้เวลาเพิ่ม"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "ลิงก์ที่ได้จากผลค้นหา"},
                },
                "required": ["url"],
            },
        },
    },
]


NOTE_MAX_CHARS = 900     # ความยาวผลค้นที่เก็บไว้เป็นความจำข้ามเทิร์น


class ToolRunner:
    """รันเครื่องมือให้โมเดล พร้อมเก็บแหล่งที่มาไว้แสดงให้ผู้ใช้เห็น"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sources: list[dict] = []      # {title, url} ที่ใช้ไปในเทิร์นนี้
        self.notes: list[str] = []         # สรุปสิ่งที่ค้นเจอ ไว้ใช้เป็นความจำเทิร์นถัดไป

    def reset(self) -> None:
        self.sources = []
        self.notes = []

    def _remember(self, items: list[dict]) -> list[dict]:
        """เก็บแหล่งที่มาไว้โชว์ พร้อมชื่อโดเมนที่อ่านออก

        ต้องแนบ host มาจากฝั่งนี้ เพราะ URL.hostname ของเบราว์เซอร์คืนโดเมนไทย
        เป็น punycode (ทองคําราคา.com → xn--42cah7d0cxcvbbb9x.com)
        """
        have = {s["url"] for s in self.sources}
        fresh = []
        for item in items:
            if item["url"] in have:
                continue
            have.add(item["url"])
            fresh.append({**item, "host": websearch.host_of(item["url"])})
        self.sources.extend(fresh)
        return fresh

    def run(self, name: str, args: dict) -> str:
        out = self._run(name, args)
        self.notes.append(f"{describe(name, args)}\n{out[:NOTE_MAX_CHARS]}")
        return out

    def _run(self, name: str, args: dict) -> str:
        try:
            if name == "web_search":
                results = websearch.search(
                    str(args.get("query", "")),
                    limit=self.cfg.search_results,
                    timeout=self.cfg.search_timeout,
                )
                self._remember(results)
                return websearch.format_results(results)

            if name == "open_page":
                url = str(args.get("url", ""))
                text = websearch.fetch_page(
                    url, max_chars=self.cfg.fetch_max_chars,
                    timeout=self.cfg.search_timeout,
                )
                self._remember([{"title": text.split("\n", 1)[0][:120], "url": url}])
                return text

            return f"ไม่รู้จักเครื่องมือชื่อ {name}"
        except websearch.SearchError as exc:
            return f"ใช้เครื่องมือไม่สำเร็จ: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ใช้เครื่องมือไม่สำเร็จ: {exc!r}"


def describe(name: str, args: dict) -> str:
    """ข้อความสั้น ๆ บอกผู้ใช้ว่ากำลังทำอะไรอยู่"""
    if name == "web_search":
        return f"กำลังค้นข้อมูล: {args.get('query', '')}"
    if name == "open_page":
        return f"กำลังเปิดอ่าน: {args.get('url', '')}"
    return f"กำลังใช้เครื่องมือ {name}"
