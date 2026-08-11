#!/usr/bin/env python3
"""ทดสอบหน้าเว็บด้วยเบราว์เซอร์จริง (Playwright + Chromium)

ใช้ไมโครโฟนปลอมของ Chromium ที่ป้อนเสียงจากไฟล์ WAV ได้
(--use-file-for-fake-audio-capture) จึงทดสอบเส้นทางเสียงจริงได้ทั้งเส้น:

    ไฟล์ WAV → getUserMedia → AudioWorklet → WebSocket → VAD → ASR → LLM
             → TTS → WebSocket → AudioBufferSource → ลำโพง (นับว่าเล่นจบจริง)

    python3 tests/test_browser.py            (headless)
    python3 tests/test_browser.py --headed   (เปิดหน้าต่างให้ดูด้วยตา)
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn                                              # noqa: E402
from playwright.sync_api import sync_playwright             # noqa: E402

from vc.api import ApiClient, pcm16_to_wav                  # noqa: E402
from vc.config import load_config                           # noqa: E402

SR = 48000
OUT = Path(__file__).resolve().parent / "_fakemic"
PASS, FAIL, SKIP = 0, 0, 0

# ตารางเวลาของเสียงที่ป้อนเข้าไมค์ปลอม (วินาที นับจากตอนเปิดไมค์)
T_QUESTION = 9.0
T_BARGE = 21.0
T_TOTAL = 32.0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✓ {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  ~ {name}  — ข้าม: {why}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, port: int):
        cfg = uvicorn.Config("web.server:app", host="127.0.0.1", port=port,
                             log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "Server":
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.1)
        raise RuntimeError("เซิร์ฟเวอร์ไม่ขึ้น")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


# ──────────────────────────────────────────────── สร้างไฟล์เสียงสำหรับไมค์ปลอม
def hiss(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.normal(0, 0.0007, int(SR * seconds)) * 32767).astype(np.int16)


def norm(pcm: np.ndarray, peak: float = 0.35) -> np.ndarray:
    top = int(np.abs(pcm).max()) or 1
    return (pcm.astype(np.float32) * (peak * 32767 / top)).astype(np.int16)


def build_mic_files(cfg) -> tuple[Path, Path, float, float]:
    """ทำ WAV 2 ไฟล์: ไฟล์เงียบ (ใช้ตอนทดสอบ UI) และไฟล์บทสนทนาตามตารางเวลา"""
    OUT.mkdir(exist_ok=True)
    quiet = OUT / "quiet.wav"
    quiet.write_bytes(pcm16_to_wav(hiss(40), SR))

    api = ApiClient(cfg)
    try:
        q = norm(api.synthesize(
            "ช่วยเล่าประวัติการสำรวจดาวอังคารแบบละเอียดหน่อยครับ", SR))
        b = norm(api.synthesize("เดี๋ยวก่อนครับ พอแค่นี้ก่อน", SR))
    finally:
        api.close()

    track = hiss(T_TOTAL)
    at_q = int(T_QUESTION * SR)
    at_b = int(T_BARGE * SR)
    track[at_q:at_q + q.size] = q[:track.size - at_q]
    track[at_b:at_b + b.size] = b[:track.size - at_b]

    convo = OUT / "convo.wav"
    convo.write_bytes(pcm16_to_wav(track, SR))
    return quiet, convo, q.size / SR, b.size / SR


def chrome_args(wav: Path) -> list[str]:
    return [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav}%noloop",
        "--autoplay-policy=no-user-gesture-required",
    ]


def watch(page) -> list[str]:
    """เก็บ error จาก console ของหน้าเว็บไว้ตรวจตอนท้าย"""
    errs: list[str] = []
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errs.append(repr(e)))
    return errs


# ──────────────────────────────────────────────────────────── ช่วงที่ 1: คู่มือ
def phase_ui(pw, port: int, quiet: Path, headed: bool) -> None:
    print("\n[1] คู่มือเริ่มต้น (onboarding) บนเบราว์เซอร์จริง")
    browser = pw.chromium.launch(headless=not headed, args=chrome_args(quiet))
    ctx = browser.new_context(permissions=["microphone"])
    page = ctx.new_page()
    errs = watch(page)
    page.goto(f"http://127.0.0.1:{port}/", wait_until="load")

    overlay = page.locator("#overlay")
    check("คู่มือเด้งขึ้นเองตอนเปิดครั้งแรก", overlay.is_visible())
    check("มีตัวบอกความคืบหน้า 5 ขั้น", page.locator("#tour-steps i").count() == 5)
    check("ขั้นแรกอธิบายภาพรวมระบบ",
          "ระบบคุยกับ AI ด้วยเสียง" in page.locator("#tour-title").inner_text())
    check("มีแผนภาพเส้นทางเสียง 5 กล่อง",
          page.locator(".flow .node").count() == 5)

    # --- ขั้นที่ 1: ไมโครโฟน ---
    page.click("#tour-next")
    check("ไปขั้นไมโครโฟนได้", page.locator("section[data-step='1']").is_visible())
    page.click("#btn-mic")
    page.wait_for_function(
        "() => document.getElementById('btn-mic').textContent.includes('พร้อม')",
        timeout=15000)
    check("ขอสิทธิ์ไมค์แล้วเปิดสตรีมได้จริง",
          "✓" in page.locator("#btn-mic").inner_text())
    check("สถานะไมค์เปลี่ยนเป็นเปิดแล้ว",
          "เปิดแล้ว" in page.locator("#mic-state").inner_text())
    sr = page.evaluate("() => window.__vl && window.__vl.sr !== undefined")
    check("ฝัง AudioContext + AudioWorklet สำเร็จ (ไม่มี error)", sr is True)

    # มิเตอร์ต้องขยับตามเสียง (ไฟล์เงียบมาก จึงเช็กแค่ว่ามันอัปเดตค่าได้)
    page.wait_for_timeout(600)
    width = page.evaluate("() => document.getElementById('mic-fill').style.width")
    check("มิเตอร์ทดสอบไมค์ทำงาน", width != "", f"width={width or '(ว่าง)'}")

    # --- ขั้นที่ 2: ตรวจระบบ ---
    page.click("#tour-next")
    page.click("#btn-check")
    page.wait_for_function(
        "() => document.querySelectorAll('#check .check-row').length >= 3",
        timeout=90000)
    rows = page.locator("#check .check-row")
    names = [rows.nth(i).locator(".name").inner_text() for i in range(rows.count())]
    oks = [rows.nth(i).get_attribute("data-ok") for i in range(rows.count())]
    check("ตรวจระบบครบ 3 ขั้น TTS/ASR/LLM", names == ["TTS", "ASR", "LLM"], str(names))
    check("ทุกขั้นผ่าน", all(o == "1" for o in oks), str(oks))
    asr_text = rows.nth(1).locator(".detail").inner_text()
    check("ขั้น ASR ถอดข้อความภาษาไทยออกมาได้", "สวัสดี" in asr_text, asr_text[:60])

    # --- ขั้นที่ 3-4 ---
    page.click("#tour-next")
    check("ขั้นสอนวิธีคุยแสดงผล", page.locator("section[data-step='3']").is_visible())
    page.click("#tour-next")
    check("ขั้นสุดท้ายสอนพูดแทรก", page.locator("section[data-step='4']").is_visible())
    check("ปุ่มสุดท้ายเปลี่ยนเป็น “เริ่มใช้งาน”",
          page.locator("#tour-next").inner_text().strip() == "เริ่มใช้งาน")

    page.click("#tour-prev")
    check("ย้อนกลับได้", page.locator("section[data-step='3']").is_visible())

    check("ไม่มี error ใน console", not errs, "; ".join(errs[:2]))
    ctx.close()
    browser.close()


# ─────────────────────────────────────────────────────── ช่วงที่ 2: คุยจริง
def phase_talk(pw, port: int, convo: Path, headed: bool) -> None:
    print("\n[2] คุยด้วยเสียงจริงผ่านเบราว์เซอร์ (ไมค์ปลอมป้อนไฟล์เสียง)")
    browser = pw.chromium.launch(headless=not headed, args=chrome_args(convo))
    ctx = browser.new_context(permissions=["microphone"])
    ctx.add_init_script(
        "localStorage.setItem('voicelink.tour.v1','1')")   # ข้ามคู่มือ
    page = ctx.new_page()
    errs = watch(page)
    page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
    check("เปิดซ้ำแล้วคู่มือไม่เด้งอีก", not page.locator("#overlay").is_visible())

    page.click("#btn-start")
    t0 = time.time()

    page.wait_for_function(
        "() => document.getElementById('link-text').textContent.includes('เชื่อมต่อแล้ว')",
        timeout=30000)
    check("เชื่อมต่อ WebSocket ได้", True,
          f"{time.time() - t0:.1f}s หลังกดเริ่ม")
    page.wait_for_selector("#chip-llm", timeout=10000)
    check("แสดงชื่อโมเดลบนแถบหัว",
          page.locator("#chip-asr").inner_text() not in ("", "—"),
          page.locator("#chip-asr").inner_text())

    # --- ทักทาย ---
    page.wait_for_selector(".msg[data-role='assistant']", timeout=40000)
    greet = page.locator(".msg[data-role='assistant'] .body").first.inner_text()
    check("AI ทักทายขึ้นเป็นข้อความ", "สวัสดี" in greet, greet[:50])

    page.wait_for_function("() => window.__vl.chunks > 0", timeout=40000)
    check("เบราว์เซอร์ได้รับเสียง TTS มาเล่นจริง", True,
          f"{page.evaluate('() => window.__vl.chunks')} ก้อน")
    page.wait_for_function("() => window.__vl.played > 0", timeout=40000)
    check("เล่นเสียงจนจบก้อนได้จริง (AudioBufferSource ทำงาน)", True,
          f"played={page.evaluate('() => window.__vl.played')}")

    # --- คำถามจากไมค์ ---
    page.wait_for_selector(".msg[data-role='user']", timeout=int(
        max(5, T_QUESTION + 25 - (time.time() - t0)) * 1000))
    user_text = page.locator(".msg[data-role='user'] .body").first.inner_text()
    check("ถอดเสียงที่พูดใส่ไมค์ปลอมเป็นข้อความไทยได้",
          "ดาวอังคาร" in user_text, repr(user_text[:70]))

    page.wait_for_function(
        "() => document.querySelectorAll(\".msg[data-role='assistant']\").length >= 2",
        timeout=60000)
    page.wait_for_timeout(1500)
    answer = page.locator(".msg[data-role='assistant'] .body").nth(1).inner_text()
    check("AI ตอบคำถามกลับมา", len(answer) > 10, repr(answer[:70]))

    check("โชว์ latency ของ ASR บนแผงข้าง",
          page.locator("#stat-asr").inner_text().endswith("ms"),
          page.locator("#stat-asr").inner_text())
    check("โชว์เวลาถึง token แรกของ LLM",
          page.locator("#stat-llm").inner_text().endswith("ms"),
          page.locator("#stat-llm").inner_text())

    # --- พูดแทรก ---
    wait_left = T_BARGE - (time.time() - t0)
    if wait_left > 0:
        page.wait_for_timeout(int(wait_left * 1000))
    speaking = page.evaluate(
        "() => ({orb: document.getElementById('orb').dataset.state,"
        " chunks: window.__vl.chunks, played: window.__vl.played})")
    still = speaking["chunks"] > speaking["played"] or speaking["orb"] == "speaking"

    if not still:
        skip("พูดแทรกกลางประโยค", "AI พูดจบก่อนถึงจังหวะพูดแทรกในไฟล์เสียง")
    else:
        try:
            page.wait_for_function("() => window.__vl.stops > 0", timeout=20000)
            vl = page.evaluate("() => window.__vl")
            check("พูดใส่ไมค์ตอน AI พูดอยู่ → สั่งหยุดเสียงถึงเบราว์เซอร์", True,
                  f"ตัดเสียงที่จองคิวไว้ {vl['cut']} ครั้ง"
                  if vl["cut"] else "จังหวะที่ขัดอยู่ระหว่างก้อนเสียงพอดี")
        except Exception:
            check("พูดใส่ไมค์ตอน AI พูดอยู่ → สั่งหยุดเสียงถึงเบราว์เซอร์", False,
                  "ไม่มีคำสั่งหยุดภายใน 20 วินาที")
        page.wait_for_timeout(2500)
        check("ติดป้าย “ถูกพูดขัด” ให้คำตอบที่ถูกขัด",
              page.locator(".badge").count() > 0,
              f"{page.locator('.badge').count()} ป้าย")
        check("นับจำนวนครั้งที่ถูกพูดแทรกบนแผงข้าง",
              page.locator("#stat-int").inner_text() != "0",
              page.locator("#stat-int").inner_text())

    # --- ตัดเสียงกลางคันแบบกำหนดจังหวะได้แน่นอน ---
    print("\n[3] สั่งหยุดตอนเสียงกำลังเล่นอยู่จริง")
    page.fill("#input", "ช่วยเล่าเรื่องดวงจันทร์ของดาวพฤหัสบดีแบบยาว ๆ ละเอียด ๆ หน่อยครับ")
    page.click("#btn-send")
    check("ส่งข้อความแบบพิมพ์ได้",
          page.wait_for_selector(".msg[data-role='user']", timeout=20000) is not None)

    # รอจนมีเสียงค้างอยู่ในคิวจริง แล้วกดหยุดใน tick เดียวกัน กันจังหวะคาบเกี่ยว
    cut = page.evaluate("""() => new Promise((resolve) => {
      const t0 = performance.now();
      const iv = setInterval(() => {
        if (window.__vl.playing() > 0) {
          const before = window.__vl.cut;
          document.getElementById('btn-stop').click();
          clearInterval(iv);
          resolve({ before, after: window.__vl.cut, waited: performance.now() - t0 });
        } else if (performance.now() - t0 > 90000) {
          clearInterval(iv);
          resolve(null);
        }
      }, 20);
    })""")
    if cut is None:
        check("กดหยุดขณะเสียงกำลังเล่น → ตัดทันที", False, "ไม่มีเสียงเล่นเลยใน 90 วินาที")
    else:
        check("กดหยุดขณะเสียงกำลังเล่น → ตัดเสียงที่จองคิวไว้ทันที",
              cut["after"] > cut["before"],
              f"cut {cut['before']} → {cut['after']} (รอ {cut['waited'] / 1000:.1f}s)")
        page.wait_for_timeout(1200)
        check("หยุดแล้วไม่มีเสียงค้างเล่นต่อ",
              page.evaluate("() => window.__vl.playing() === 0"),
              page.evaluate("() => `ยังเล่นอยู่ ${__vl.playing()} ก้อน`"))

    # --- ค้นเน็ตแล้วโชว์แหล่งที่มา ---
    print("\n[4] ค้นข้อมูลจากอินเทอร์เน็ตและแสดงแหล่งที่มา")
    page.fill("#input", "ราคาทองคำแท่งวันนี้รับซื้อบาทละเท่าไหร่")
    page.click("#btn-send")
    try:
        page.wait_for_function(
            "() => [...document.querySelectorAll('#logs div')]"
            ".some(d => d.textContent.includes('ค้นเน็ต'))", timeout=90000)
        check("โมเดลสั่งค้นเน็ตเองเมื่อถูกถามเรื่องปัจจุบัน", True,
              page.evaluate("() => [...document.querySelectorAll('#logs div')]"
                            ".filter(d => d.textContent.includes('ค้นเน็ต'))"
                            ".map(d => d.textContent.slice(10))[0]"))
    except Exception:
        check("โมเดลสั่งค้นเน็ตเองเมื่อถูกถามเรื่องปัจจุบัน", False,
              "ไม่มีการค้นภายใน 90 วินาที")

    try:
        page.wait_for_selector(".sources a", timeout=90000)
        hosts = page.evaluate(
            "() => [...document.querySelectorAll('.sources a')].map(a => a.textContent)")
        check("แสดงแหล่งที่มาใต้คำตอบเป็นลิงก์กดได้", len(hosts) > 0,
              ", ".join(hosts[:4]))
        check("ลิงก์แหล่งที่มาเปิดแท็บใหม่อย่างปลอดภัย",
              page.evaluate("() => { const a = document.querySelector('.sources a');"
                            " return a.target === '_blank'"
                            " && a.rel.includes('noopener')"
                            " && a.href.startsWith('http'); }"))
    except Exception:
        check("แสดงแหล่งที่มาใต้คำตอบเป็นลิงก์กดได้", False, "ไม่มี .sources ขึ้นมา")

    page.wait_for_timeout(2000)
    answer = page.evaluate(
        "() => { const m = document.querySelectorAll(\".msg[data-role='assistant']\");"
        " return m[m.length-1].querySelector('.body').textContent; }")
    check("คำตอบมีตัวเลขราคา ไม่ใช่บอกว่าทำไม่ได้",
          any(c.isdigit() for c in answer), repr(answer[:80]))

    # --- ปุ่มควบคุม ---
    print("\n[5] ปุ่มควบคุมบนหน้าเว็บ")
    page.click("#btn-mute")
    check("ปุ่มปิดเสียงสลับสถานะได้",
          page.locator("#btn-mute").get_attribute("data-on") == "1",
          page.locator("#btn-mute").inner_text())
    page.click("#btn-mute")
    check("กดซ้ำแล้วเปิดเสียงกลับ",
          page.locator("#btn-mute").get_attribute("data-on") == "0")

    page.click("#btn-echo")
    check("ปุ่มโหมดลำโพง/หูฟังสลับได้",
          page.locator("#btn-echo").get_attribute("data-on") == "1")

    before = page.locator(".msg").count()
    page.click("#btn-clear")
    page.wait_for_function("() => document.querySelectorAll('.msg').length === 0",
                           timeout=10000)
    check("ปุ่มล้างประวัติเคลียร์บทสนทนา", before > 0, f"ลบไป {before} ข้อความ")

    page.click("#btn-help")
    check("ปุ่ม “? วิธีใช้” เปิดคู่มือซ้ำได้", page.locator("#overlay").is_visible())

    check("ไม่มี error ใน console ตลอดการคุย", not errs, "; ".join(errs[:2]))
    ctx.close()
    browser.close()


def main() -> int:
    headed = "--headed" in sys.argv
    cfg = load_config()
    print("กำลังสร้างไฟล์เสียงสำหรับไมโครโฟนปลอม (เรียก TTS จริง)...")
    quiet, convo, q_s, b_s = build_mic_files(cfg)
    print(f"  {convo.name}: คำถามที่วินาทีที่ {T_QUESTION:.0f} ({q_s:.1f}s) · "
          f"พูดแทรกที่วินาทีที่ {T_BARGE:.0f} ({b_s:.1f}s) · ยาวรวม {T_TOTAL:.0f}s")

    port = free_port()
    with Server(port), sync_playwright() as pw:
        phase_ui(pw, port, quiet, headed)
        phase_talk(pw, port, convo, headed)

    print()
    tail = f" · ข้าม {SKIP}" if SKIP else ""
    if FAIL:
        print(f"ล้มเหลว {FAIL} ข้อ (ผ่าน {PASS}{tail}) ✗")
        return 1
    print(f"ผ่านทั้งหมด {PASS} ข้อ{tail} ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
