#!/usr/bin/env python3
"""ทดสอบการค้นข้อมูลจากอินเทอร์เน็ต

ส่วน [1]-[3] ไม่ต่อเน็ต (ทดสอบตัวแยกผลลัพธ์และการกันยิงเข้าเครือข่ายใน)
ส่วน [4]-[5] ต่อเน็ตจริง ข้ามได้ด้วย --offline
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vc.api import ApiClient, _merge_tool_deltas      # noqa: E402
from vc.config import load_config                     # noqa: E402
from vc.tools import TOOLS, ToolRunner                # noqa: E402
from vc.websearch import (                            # noqa: E402
    SearchError, _ResultParser, format_results, fetch_page, html_to_text,
    search, sources,
)

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✓ {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


SAMPLE = """
<div class="result results_links web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://example.com/gold">ราคาทอง&amp;แนวโน้ม</a>
  </h2>
  <a class="result__snippet" href="x">ราคาทองวันนี้ <b>4,363</b> บาท ตามประกาศสมาคม</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnews.example.org%2Fa&amp;rut=9">ข่าวเช้า</a>
  </h2>
  <a class="result__snippet">สรุปข่าววันนี้</a>
</div>
<div class="result results_links result--ad">
  <h2><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">โฆษณา</a></h2>
</div>
<div class="result">
  <h2><a class="result__a" href="https://example.com/second">หน้าที่สองของเว็บเดิม</a></h2>
  <a class="result__snippet">ซ้ำโดเมน</a>
</div>
"""


def test_parser() -> None:
    print("\n[1] แยกผลค้นหาจากหน้า HTML")
    p = _ResultParser()
    p.feed(SAMPLE)
    r = p.results
    check("ได้ผลลัพธ์ที่ใช้ได้ 3 รายการ (ตัดโฆษณาออก)", len(r) == 3, f"got={len(r)}")
    check("ถอด HTML entity ในหัวข้อ", r[0]["title"] == "ราคาทอง&แนวโน้ม", r[0]["title"])
    check("ดึงคำอธิบายย่อรวมตัวหนาได้", "4,363" in r[0]["snippet"], r[0]["snippet"][:50])
    check("คลี่ลิงก์เปลี่ยนทางของ DuckDuckGo",
          r[1]["url"] == "https://news.example.org/a", r[1]["url"])
    check("ตัดลิงก์โฆษณาของ DuckDuckGo ทิ้ง",
          all("duckduckgo.com" not in x["url"] for x in r))

    from vc.websearch import _ResultParser as RP
    p2 = RP()
    p2.feed(SAMPLE)
    urls = [x["url"] for x in p2.results]
    check("ยังไม่ตัดโดเมนซ้ำในชั้น parser", urls.count("https://example.com/second") == 1)

    check("format_results จัดข้อความให้โมเดลอ่านได้",
          "[1]" in format_results(r) and "ที่มา:" in format_results(r))
    check("สรุปชื่อโดเมนสำหรับโชว์ผู้ใช้",
          sources(r) == "example.com, news.example.org", sources(r))
    check("ไม่มีผล → บอกตรง ๆ ไม่ปล่อยว่าง", format_results([]) == "ไม่พบผลการค้นหา")


def test_text_extract() -> None:
    print("\n[2] ดึงข้อความจากหน้าเว็บ")
    html = """<html><head><title>หัวเรื่อง</title>
      <style>.a{color:red}</style><script>var x=1;alert("ห้ามเอามา")</script></head>
      <body><nav>เมนู ห้ามเอามา</nav>
      <h1>ราคาทอง</h1><p>วันนี้ราคา <b>4,363</b> บาท</p>
      <p>อัปเดตล่าสุด</p><footer>ลิขสิทธิ์ ห้ามเอามา</footer></body></html>"""
    title, text = html_to_text(html)
    check("ได้ชื่อหน้า", title == "หัวเรื่อง", title)
    check("ได้เนื้อหาหลัก", "4,363" in text and "ราคาทอง" in text)
    check("ตัด script/style/nav/footer ออก", "ห้ามเอามา" not in text,
          repr(text[:80]))
    check("ไม่มีบรรทัดว่างซ้อนเกินสองบรรทัด", "\n\n\n" not in text)


def test_guards() -> None:
    print("\n[3] กันเปิดที่อยู่ที่ไม่ควรเปิด")
    for url, why in [
        ("file:///etc/passwd", "ไม่ใช่ http"),
        ("http://127.0.0.1:8000/", "loopback"),
        ("http://192.168.1.1/", "วง LAN"),
        ("http://[::1]/", "loopback v6"),
    ]:
        try:
            fetch_page(url)
            check(f"บล็อก {why}", False, "เปิดผ่าน (ไม่ควร)")
        except SearchError:
            check(f"บล็อก {why}", True, url)
        except Exception as exc:  # noqa: BLE001
            check(f"บล็อก {why}", False, repr(exc)[:60])

    print("\n   ประกอบ tool_call ที่สตรีมมาเป็นชิ้น ๆ")
    acc: dict = {}
    _merge_tool_deltas(acc, [{"index": 0, "id": "c1",
                              "function": {"name": "web_search", "arguments": '{"qu'}}])
    _merge_tool_deltas(acc, [{"index": 0, "function": {"arguments": 'ery":"ทอง"}'}}])
    check("ต่ออาร์กิวเมนต์ที่มาเป็นชิ้น ๆ ได้ครบ",
          acc[0] == {"id": "c1", "name": "web_search", "args": '{"query":"ทอง"}'},
          str(acc[0]))
    acc2: dict = {}
    _merge_tool_deltas(acc2, [{"id": "z", "function": {"name": "web_search",
                                                       "arguments": "{}"}}])
    check("ไม่มี index ก็ยังประกอบได้", acc2[0]["name"] == "web_search")


def test_live_search() -> None:
    print("\n[4] ค้นจากอินเทอร์เน็ตจริง")
    try:
        res = search("ราคาทองคำวันนี้", limit=5)
    except SearchError as exc:
        check("ค้นได้", False, str(exc))
        return
    check("ได้ผลค้นหากลับมา", len(res) >= 3, f"{len(res)} ผล")
    check("ทุกผลมีลิงก์ http(s)",
          all(r["url"].startswith("http") for r in res))
    check("ทุกผลมีหัวข้อ", all(r["title"] for r in res))
    check("ส่วนใหญ่มีคำอธิบายย่อ",
          sum(1 for r in res if r["snippet"]) >= len(res) - 1)
    check("ไม่ซ้ำโดเมน", len({r["url"].split("/")[2] for r in res}) == len(res))
    print(f"      แหล่ง: {sources(res, limit=5)}")


def test_end_to_end() -> None:
    print("\n[5] ให้โมเดลค้นเน็ตแล้วตอบด้วยข้อมูลปัจจุบัน")
    cfg = load_config()
    api = ApiClient(cfg)
    runner = ToolRunner(cfg)
    used: list[str] = []

    def run_tool(name: str, args: dict) -> str:
        used.append(f"{name}({args.get('query') or args.get('url')})")
        return runner.run(name, args)

    try:
        msgs = [cfg.system_message(),
                {"role": "user", "content": "ราคาทองคำแท่งวันนี้รับซื้อบาทละเท่าไหร่"}]
        out = ""
        for delta in api.chat_stream(msgs, threading.Event(), tools=TOOLS,
                                     run_tool=run_tool, max_rounds=2):
            out += delta
    except Exception as exc:  # noqa: BLE001
        check("เรียกโมเดลพร้อมเครื่องมือได้", False, repr(exc)[:150])
        api.close()
        return
    api.close()

    check("โมเดลเลือกใช้เครื่องมือค้นเน็ตเอง", bool(used), " · ".join(used)[:90])
    check("ตอบกลับมาเป็นข้อความ", len(out.strip()) > 10, repr(out.strip()[:90]))
    check("เก็บแหล่งที่มาไว้แสดงผู้ใช้", bool(runner.sources),
          sources(runner.sources))
    check("คำตอบมีตัวเลข (ไม่ใช่บอกว่าทำไม่ได้)",
          any(c.isdigit() for c in out), repr(out.strip()[:110]))
    check("ไม่อ่าน URL ออกมาในคำตอบ", "http" not in out.lower(),
          repr(out.strip()[:80]))

    # ต้องไม่ค้นพร่ำเพรื่อ ไม่งั้นคำถามธรรมดาก็ช้าไปด้วย
    print("\n[6] คำถามความรู้ทั่วไปต้องตอบเองโดยไม่ค้น")
    api2 = ApiClient(cfg)
    used2: list[str] = []
    out2 = ""
    try:
        for delta in api2.chat_stream(
                [cfg.system_message(),
                 {"role": "user", "content": "น้ำเดือดที่กี่องศาเซลเซียส"}],
                threading.Event(), tools=TOOLS,
                run_tool=lambda n, a: used2.append(n) or "", max_rounds=2):
            out2 += delta
    except Exception as exc:  # noqa: BLE001
        check("ถามความรู้ทั่วไปได้", False, repr(exc)[:120])
    finally:
        api2.close()
    check("ไม่เสียเวลาค้นเน็ตกับคำถามที่ไม่ต้องค้น", not used2, str(used2))
    check("ยังตอบคำถามความรู้ทั่วไปได้", "100" in out2, repr(out2.strip()[:80]))
    check("ยังไม่มี Markdown ในคำตอบ (กฎเดิมยังอยู่)",
          not any(t in out2 for t in ("**", "##", "- ")), repr(out2.strip()[:60]))


def main() -> int:
    test_parser()
    test_text_extract()
    test_guards()
    if "--offline" in sys.argv:
        print("\n(ข้ามส่วนที่ต้องต่อเน็ต)")
    else:
        test_live_search()
        test_end_to_end()
    print()
    if FAIL:
        print(f"ล้มเหลว {FAIL} ข้อ (ผ่าน {PASS}) ✗")
        return 1
    print(f"ผ่านทั้งหมด {PASS} ข้อ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
