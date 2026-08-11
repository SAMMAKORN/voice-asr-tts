"""ค้นข้อมูลจากอินเทอร์เน็ตให้ LLM ใช้ตอบเรื่องปัจจุบัน

ใช้ DuckDuckGo (endpoint หน้า HTML) เพราะไม่ต้องสมัคร API key
แยกผลลัพธ์ด้วย html.parser ของ stdlib จึงไม่ต้องเพิ่ม dependency

  search("ราคาทองวันนี้")  → [{title, url, snippet}, ...]
  fetch_page(url)          → เนื้อหาหน้าเว็บเป็นข้อความล้วน
"""
from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

ENDPOINT = "https://html.duckduckgo.com/html/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
_WS = re.compile(r"[ \t\xa0]+")
_NL = re.compile(r"\n{3,}")


class SearchError(RuntimeError):
    pass


# ────────────────────────────────────────────────────── แยกผลลัพธ์จากหน้า DuckDuckGo
class _ResultParser(HTMLParser):
    """เก็บคู่ (ลิงก์+หัวข้อ) กับ (คำอธิบายย่อ) ตามลำดับที่ปรากฏในหน้า"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._grab = ""          # "title" | "snippet" | ""
        self._buf: list[str] = []
        self._href = ""
        self._depth = 0

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        for k, v in attrs:
            if k == "class" and v:
                return set(v.split())
        return set()

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
        for k, v in attrs:
            if k == name:
                return v or ""
        return ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._grab:
            self._depth += 1
            return
        cls = self._classes(attrs)
        if tag == "a" and "result__a" in cls:
            self._grab, self._buf, self._depth = "title", [], 0
            self._href = self._attr(attrs, "href")
        elif "result__snippet" in cls:
            self._grab, self._buf, self._depth = "snippet", [], 0

    def handle_endtag(self, tag: str) -> None:
        if not self._grab:
            return
        if self._depth > 0:
            self._depth -= 1
            return
        text = _WS.sub(" ", "".join(self._buf)).strip()
        if self._grab == "title":
            url = _clean_url(self._href)
            if url and text:
                self.results.append({"title": text, "url": url, "snippet": ""})
        elif self._grab == "snippet" and self.results and not self.results[-1]["snippet"]:
            self.results[-1]["snippet"] = text
        self._grab, self._buf = "", []

    def handle_data(self, data: str) -> None:
        if self._grab:
            self._buf.append(data)


def _clean_url(href: str) -> str:
    """DuckDuckGo บางทีห่อลิงก์ไว้ในตัวเปลี่ยนทาง — ดึงปลายทางจริงออกมา"""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        href = unquote(target)
        parsed = urlparse(href)
    if parsed.scheme not in ("http", "https"):
        return ""
    if "duckduckgo.com" in parsed.netloc:      # โฆษณา/ลิงก์ภายในของ DDG
        return ""
    return href


# ───────────────────────────────────────────────────────────── ดึงข้อความจากหน้าเว็บ
class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        out = _WS.sub(" ", "".join(self.parts))
        out = "\n".join(line.strip() for line in out.split("\n"))
        return _NL.sub("\n\n", out).strip()


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextParser()
    p.feed(html)
    return p.title, p.text()


# ──────────────────────────────────────────────────────────────── กันยิงเข้าเครือข่ายใน
def _is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            return False
    return True


# ───────────────────────────────────────────────────────────────────────── API หลัก
def search(query: str, limit: int = 5, timeout: float = 12.0) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    try:
        r = httpx.post(ENDPOINT, data={"q": query, "kl": "th-th"},
                       headers={"User-Agent": UA, "Accept-Language": "th,en;q=0.8"},
                       timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise SearchError(f"ต่ออินเทอร์เน็ตไม่ได้: {exc}") from exc
    if r.status_code >= 400:
        raise SearchError(f"เครื่องค้นหาตอบ {r.status_code}")

    p = _ResultParser()
    p.feed(r.text)
    out: list[dict] = []
    seen: set[str] = set()
    for item in p.results:
        host = urlparse(item["url"]).netloc
        if host in seen:              # เอาหน้าเดียวพอต่อหนึ่งเว็บ ผลจะได้หลากหลาย
            continue
        seen.add(host)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def fetch_page(url: str, max_chars: int = 3000, timeout: float = 12.0) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SearchError("เปิดได้เฉพาะลิงก์ http/https")
    if not _is_public(parsed.hostname or ""):
        raise SearchError("ไม่อนุญาตให้เปิดที่อยู่ในเครือข่ายภายใน")
    try:
        r = httpx.get(url, headers={"User-Agent": UA}, timeout=timeout,
                      follow_redirects=True)
    except httpx.HTTPError as exc:
        raise SearchError(f"เปิดหน้าเว็บไม่ได้: {exc}") from exc
    if r.status_code >= 400:
        raise SearchError(f"หน้าเว็บตอบ {r.status_code}")
    if "html" not in r.headers.get("content-type", "").lower():
        raise SearchError("ลิงก์นี้ไม่ใช่หน้าเว็บที่อ่านเป็นข้อความได้")

    title, text = html_to_text(r.text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n…(ตัดเนื้อหาส่วนที่เหลือออก)"
    return f"{title}\n{url}\n\n{text}" if title else f"{url}\n\n{text}"


def format_results(results: list[dict]) -> str:
    """จัดผลค้นหาให้โมเดลอ่านง่าย"""
    if not results:
        return "ไม่พบผลการค้นหา"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    ที่มา: {r['url']}\n    {r['snippet']}")
    return "\n\n".join(lines)


def host_of(url: str) -> str:
    """ชื่อโดเมนที่อ่านออก — โดเมนไทยที่เป็น punycode จะถอดกลับให้"""
    host = urlparse(url).netloc.removeprefix("www.")
    if "xn--" in host:
        try:
            host = host.encode("ascii").decode("idna")
        except (UnicodeError, ValueError):
            pass
    return host


def sources(results: list[dict], limit: int = 3) -> str:
    hosts: list[str] = []
    for r in results:
        host = r.get("host") or host_of(r["url"])
        if host and host not in hosts:
            hosts.append(host)
    return ", ".join(hosts[:limit])
