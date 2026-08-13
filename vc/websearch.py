"""ค้นข้อมูลจากอินเทอร์เน็ตให้ LLM ใช้ตอบเรื่องปัจจุบัน

ใช้ DuckDuckGo (endpoint หน้า HTML) เพราะไม่ต้องสมัคร API key
แยกผลลัพธ์ด้วย html.parser ของ stdlib จึงไม่ต้องเพิ่ม dependency

  search("ราคาทองวันนี้")  → [{title, url, snippet}, ...]
  fetch_page(url)          → เนื้อหาหน้าเว็บเป็นข้อความล้วน
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

ENDPOINT = "https://html.duckduckgo.com/html/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
_WS = re.compile(r"[ \t\xa0]+")
_NL = re.compile(r"\n{3,}")

# ───────────────────────────────────── ข้อจำกัดของการเปิดหน้าเว็บ (P1-3: SSRF)
MAX_HOPS = 3                       # จำนวน redirect สูงสุดที่ยอมตาม
ALLOWED_PORTS = {80, 443}          # พอร์ตอื่นคือการสแกนพอร์ตภายในโดยปริยาย
ALLOWED_SCHEMES = ("http", "https")
DEFAULT_MAX_BYTES = 512_000        # เพดานไบต์ที่ยอมอ่านเข้ามาต่อหนึ่งหน้า
ALLOWED_CONTENT_TYPES = ("text/", "application/xhtml+xml", "application/json")
CONNECT_TIMEOUT = 5.0


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
    """ที่อยู่นี้ชี้ออกอินเทอร์เน็ตจริงไหม — ตรวจ **ทุก** IP ที่ DNS ตอบกลับมา

    known limitation: DNS rebinding TOCTOU — เราตรวจตอน resolve แล้วปล่อยให้
    httpx resolve เองอีกครั้งตอนต่อจริง ผู้โจมตีที่คุมโดเมนและตั้ง TTL สั้นมาก
    ยังสลับคำตอบระหว่างสองจังหวะได้ การอุดต้องต่อไปที่ IP ที่ตรวจแล้วโดยตรง
    พร้อมตั้ง Host header/SNI เอง ซึ่งกระทบการตรวจใบรับรอง TLS — เก็บไว้เป็น
    งานแยก (ดู PR ของ P1-3)
    """
    if not host:
        return False
    bare = host.strip("[]")
    try:                       # เป็น IP ตรง ๆ ไม่ต้องถาม DNS
        ip = ipaddress.ip_address(bare)
    except ValueError:
        pass
    else:
        return _public_ip(ip)
    try:
        infos = socket.getaddrinfo(bare, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not _public_ip(ip):
            return False
    return True


def _public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return False
    if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)     # ::ffff:127.0.0.1
    if mapped is not None:
        return _public_ip(mapped)
    return True


def assert_fetchable(url: str) -> str:
    """ตรวจว่า URL นี้ยอมให้เปิดได้ไหม — คืน URL เดิมถ้าผ่าน ไม่ผ่านโยน SearchError

    ใช้ร่วมกันทั้งตอนเปิดหน้าเว็บและตอนตรวจ URL ที่โมเดลขอเปิด (P1-4)
    กติกา: http/https เท่านั้น · พอร์ต 80/443 เท่านั้น · โฮสต์ต้องชี้ IP สาธารณะ
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
        raise SearchError("เปิดได้เฉพาะลิงก์ http/https")
    if parsed.username or parsed.password:
        raise SearchError("ไม่อนุญาตลิงก์ที่ฝังชื่อผู้ใช้/รหัสผ่าน")
    try:
        port = parsed.port
    except ValueError:
        raise SearchError("หมายเลขพอร์ตในลิงก์ไม่ถูกต้อง") from None
    # ตรวจโฮสต์ก่อนพอร์ต เพื่อให้ข้อความบอกสาเหตุที่สำคัญกว่า (ยิงเข้าวงใน)
    if not _is_public(parsed.hostname or ""):
        raise SearchError("ไม่อนุญาตให้เปิดที่อยู่ในเครือข่ายภายใน")
    port = port or (443 if parsed.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise SearchError(f"ไม่อนุญาตพอร์ต {port} (เปิดได้เฉพาะ 80 และ 443)")
    return url


def _content_type_ok(value: str) -> bool:
    kind = (value or "").split(";", 1)[0].strip().lower()
    return any(kind.startswith(ok) for ok in ALLOWED_CONTENT_TYPES)


def _max_bytes() -> int:
    try:
        return max(1024, int(os.environ.get("FETCH_MAX_BYTES") or DEFAULT_MAX_BYTES))
    except ValueError:
        return DEFAULT_MAX_BYTES


# ───────────────────────────────────────────────────────────────────────── API หลัก
def search(query: str, limit: int = 5, timeout: float = 12.0) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    try:
        r = httpx.post(ENDPOINT, data={"q": query, "kl": "th-th"},
                       headers={"User-Agent": UA, "Accept-Language": "th,en;q=0.8"},
                       timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
                       follow_redirects=True)
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


def fetch_page(url: str, max_chars: int = 3000, timeout: float = 12.0,
               max_bytes: int | None = None,
               transport: httpx.BaseTransport | None = None) -> str:
    """เปิดอ่านหน้าเว็บแบบมีขอบเขต

    เดิมเช็คแค่ URL แรกแล้วสั่ง `follow_redirects=True` ผู้โจมตีจึงพา redirect
    เข้าเครือข่ายภายในได้ (PoC สำเร็จจริง) ตอนนี้จึงตาม redirect เอง แล้วตรวจ
    ทุก hop ด้วย `assert_fetchable()` พร้อมเพดานอีกสามชั้น:
      · จำนวน hop  · ขนาดไบต์ที่อ่าน (FETCH_MAX_BYTES)  · เวลารวมทั้งคำขอ
    """
    cap = max_bytes if max_bytes is not None else _max_bytes()
    deadline = time.monotonic() + max(1.0, timeout)
    target = assert_fetchable(url)

    def left() -> float:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise SearchError(f"เปิดหน้าเว็บไม่ทันใน {timeout:.0f} วินาที")
        return remain

    limits = httpx.Timeout(connect=min(CONNECT_TIMEOUT, max(1.0, timeout)),
                           read=timeout, write=timeout, pool=timeout)
    client = httpx.Client(
        timeout=limits,
        follow_redirects=False,
        transport=transport,
        # ขอเนื้อหาดิบเท่านั้น เพื่อให้ FETCH_MAX_BYTES เป็นเพดานหน่วยความจำจริง
        # `iter_bytes()` ของ httpx จะคลาย gzip ก่อนคืน chunk ซึ่งเปิดทางให้ zip bomb
        headers={"User-Agent": UA, "Accept-Encoding": "identity"},
    )
    try:
        for hop in range(MAX_HOPS + 1):
            remain = left()
            # timeout ต้องหดตามงบรวมที่เหลือ ไม่ใช่เริ่มนับ `timeout` ใหม่ทุก redirect
            request_timeout = httpx.Timeout(
                remain, connect=min(CONNECT_TIMEOUT, remain))
            try:
                with client.stream("GET", target, timeout=request_timeout) as r:
                    if r.is_redirect:
                        location = r.headers.get("location", "")
                        if hop >= MAX_HOPS:
                            raise SearchError(
                                f"too many redirects (เกิน {MAX_HOPS} ครั้ง)")
                        if not location:
                            raise SearchError("หน้าเว็บสั่งเปลี่ยนทางแต่ไม่บอกปลายทาง")
                        # ตรวจ "ทุก hop" — จุดที่บั๊กเดิมข้ามไป
                        target = assert_fetchable(urljoin(str(r.url), location))
                        continue
                    if r.status_code >= 400:
                        raise SearchError(f"หน้าเว็บตอบ {r.status_code}")
                    # ตรวจชนิดเนื้อหาจาก header ก่อนอ่าน body แม้แต่ไบต์เดียว
                    ctype = r.headers.get("content-type", "")
                    if not _content_type_ok(ctype):
                        raise SearchError(
                            "ลิงก์นี้ไม่ใช่หน้าเว็บที่อ่านเป็นข้อความได้"
                            + (f" ({ctype.split(';')[0]})" if ctype else ""))
                    content_encoding = (
                        r.headers.get("content-encoding", "").strip().lower())
                    if content_encoding not in ("", "identity"):
                        # ตรวจ header ก่อนอ่าน body แม้แต่ไบต์เดียว: เซิร์ฟเวอร์ที่ไม่ทำตาม
                        # Accept-Encoding: identity อาจส่ง gzip bomb ที่ขยายใหญ่กว่า cap มาก
                        raise SearchError(
                            "หน้าเว็บส่งเนื้อหาแบบบีบอัดแม้ขอ identity "
                            f"({content_encoding}) จึงไม่อ่านเพื่อความปลอดภัย")
                    body = bytearray()
                    # ไม่ระบุ chunk_size: ถ้าบังคับให้สะสมก้อนใหญ่ slow-drip จะค้าง
                    # อยู่ใน iterator จน deadline ไม่มีโอกาสถูกตรวจ
                    # หลังปฏิเสธ Content-Encoding แล้ว iter_bytes() จะไม่คลายข้อมูล
                    # เพิ่ม และยังรองรับ MockTransport/response ที่ buffer มาแล้ว
                    for chunk in r.iter_bytes():
                        left()
                        room = cap - len(body)
                        if room <= 0:
                            break
                        body.extend(chunk[:room])
                        if len(chunk) >= room:
                            break           # ออกจาก with = ปิดคอนเนกชันทันที
                    html = bytes(body).decode(r.encoding or "utf-8", errors="replace")
                    break
            except httpx.HTTPError as exc:
                raise SearchError(f"เปิดหน้าเว็บไม่ได้: {exc}") from exc
        else:
            raise SearchError(f"too many redirects (เกิน {MAX_HOPS} ครั้ง)")
    finally:
        client.close()

    title, text = html_to_text(html)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n…(ตัดเนื้อหาส่วนที่เหลือออก)"
    return f"{title}\n{target}\n\n{text}" if title else f"{target}\n\n{text}"


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
    """สรุปชื่อโดเมนของผลค้นหา ใช้โชว์แหล่งที่มาแบบย่อ"""
    hosts: list[str] = []
    for r in results:
        host = r.get("host") or host_of(r["url"])
        if host and host not in hosts:
            hosts.append(host)
    return ", ".join(hosts[:limit])
