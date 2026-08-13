"""P2-7 … P2-14 — พฤติกรรมจริงบน Chromium (Playwright)

    pytest -m browser tests/test_frontend_ui.py

ไม่แตะ API จริงเลย: เซิร์ฟเวอร์ปลอมในไฟล์นี้เสิร์ฟไฟล์ static ชุดจริงและพูด
โปรโตคอล WebSocket เดียวกับ web/server.py เท่าที่หน้าเว็บต้องใช้ จึงทดสอบได้ทั้ง
การวาด DOM, การเชื่อมต่อใหม่เมื่อเซิร์ฟเวอร์ดับ และข้อความ error ของไมโครโฟน
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
# ต้อง import ที่ระดับโมดูล: ไฟล์นี้ใช้ `from __future__ import annotations` ทำให้
# FastAPI ต้องหาชนิดของพารามิเตอร์จาก globals ของโมดูล ถ้า import ไว้ในฟังก์ชัน
# มันจะแก้ชื่อ `WebSocket` ไม่ออกแล้วปฏิเสธการเชื่อมต่อด้วย HTTP 403 เงียบ ๆ
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

pytestmark = [pytest.mark.browser]

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"
TOKEN = "เทสต์-token"


# ──────────────────────────────────────────────────────── เซิร์ฟเวอร์ปลอม
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServer:
    """เสิร์ฟหน้าเว็บจริง + /ws ที่ส่ง ready ให้ แล้วปิด/เปิดได้ตามใจเทสต์"""

    def __init__(self, port: int) -> None:
        self.port = port
        self.sessions = 0
        self._server = None
        self._thread: threading.Thread | None = None

    def _app(self):
        app = FastAPI()
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        async def index() -> HTMLResponse:
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(html.replace("__SESSION_TOKEN__", TOKEN))

        @app.get("/api/config")
        async def config() -> JSONResponse:
            return JSONResponse({"chat_model": "fake-chat", "asr_model": "fake-asr",
                                 "tts_model": "fake-tts", "mic_sr": 16000,
                                 "speaker_sr": 24000, "frame_ms": 20})

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            self.sessions += 1
            await ws.send_json({
                "type": "ready", "chat_model": "fake-chat", "asr_model": "fake-asr",
                "tts_model": "fake-tts", "mic_sr": 16000, "speaker_sr": 24000,
                "frame_ms": 20, "tts_enabled": True, "echo_guard": True,
                "web_search": False, "api_configured": True,
            })
            try:
                while True:
                    await ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                return

        return app

    def start(self) -> None:
        import uvicorn

        cfg = uvicorn.Config(self._app(), host="127.0.0.1", port=self.port,
                             log_level="error")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError("เซิร์ฟเวอร์ปลอมไม่ขึ้น")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(10)
        self._server = self._thread = None


@pytest.fixture(scope="module")
def server():
    srv = FakeServer(free_port())
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                    "--use-fake-device-for-media-stream"])
        try:
            yield b
        finally:
            b.close()


def open_page(browser, server, *, width: int = 1440, height: int = 900,
              init: str = "", reduced: bool = False):
    page = browser.new_page(viewport={"width": width, "height": height})
    if reduced:
        page.emulate_media(reduced_motion="reduce")
    page.add_init_script("localStorage.setItem('voicelink.tour.v1','1')")
    if init:
        page.add_init_script(init)
    page.goto(f"http://127.0.0.1:{server.port}/", wait_until="load")
    page.wait_for_function("typeof handleJson === 'function'")
    return page


def orb_state(page) -> str:
    return page.get_attribute("#orb", "data-state")


def dashoffset(page) -> float:
    return float(page.get_attribute("#orb-val", "stroke-dashoffset"))


# ─────────────────────────────────────────────────────────── P2-7 orb
def test_orb_ring_follows_the_voice_level(browser, server) -> None:
    """AC-7.1 — stroke-dashoffset ถูกเขียนจริงและขยับตามระดับเสียง"""
    page = open_page(browser, server)
    page.evaluate("setOrb('listening')")
    page.evaluate("handleJson({type:'level', rms:0.0, threshold:0.02})")
    page.wait_for_timeout(250)
    quiet = dashoffset(page)
    page.evaluate("handleJson({type:'level', rms:0.30, threshold:0.02})")
    page.wait_for_timeout(400)
    loud = dashoffset(page)
    assert loud < quiet - 20, f"วงแหวนไม่ขยับตามเสียง ({quiet} → {loud})"
    assert page.evaluate("window.__vl.orbWrites") > 3
    page.close()


def test_search_state_uses_its_own_colour(browser, server) -> None:
    """AC-7.2"""
    page = open_page(browser, server)
    page.evaluate("handleJson({type:'status', icon:'🔎', text:'ค้นข้อมูล'})")
    assert orb_state(page) == "searching"
    violet = page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--violet').trim()")
    stroke = page.evaluate("getComputedStyle($('orb-val')).stroke")
    assert violet and stroke != "none"
    page.close()


def test_unknown_icon_is_thinking_not_idle(browser, server) -> None:
    """AC-7.3"""
    page = open_page(browser, server)
    page.evaluate("handleJson({type:'status', icon:'🛰', text:'อะไรก็ไม่รู้'})")
    assert orb_state(page) == "thinking"
    page.close()


def test_thinking_state_animates_and_reduced_motion_disables_it(browser, server) -> None:
    """AC-7.4 / AC-7.5"""
    page = open_page(browser, server)
    page.evaluate("handleJson({type:'status', icon:'💭', text:'กำลังคิด'})")
    name = page.evaluate("getComputedStyle($('orb').querySelector('svg.ring')).animationName")
    assert name == "orb-sweep"
    page.close()

    calm = open_page(browser, server, reduced=True)
    calm.evaluate("handleJson({type:'status', icon:'💭', text:'กำลังคิด'})")
    assert calm.evaluate(
        "getComputedStyle($('orb').querySelector('svg.ring')).animationName") == "none"
    assert orb_state(calm) == "thinking", "ยังต้องอ่านสถานะได้จากสี/ข้อความ"
    calm.close()


# ──────────────────────────────────────────────────── P2-8 error บนมือถือ
def test_error_is_visible_on_a_phone_viewport(browser, server) -> None:
    """AC-8.1 / AC-8.4 — iPhone 390px ต้องเห็น error โดยไม่ต้องเลื่อนจอ"""
    page = open_page(browser, server, width=390, height=844)
    assert page.is_hidden(".side"), "แผงข้างควรถูกซ่อนบนจอเล็ก (สภาพเดิมของ CSS)"
    page.evaluate("handleJson({type:'log', level:'error', text:'ถอดเสียงไม่สำเร็จ'})")
    alert = page.locator("#alert")
    alert.wait_for(state="visible", timeout=2000)
    assert "ถอดเสียงไม่สำเร็จ" in alert.inner_text()
    box = alert.bounding_box()
    assert box and box["y"] + box["height"] <= 844 + 1, "ต้องเห็นได้ไม่ต้องเลื่อนจอ"
    assert orb_state(page) == "error"
    assert page.get_attribute("#alert", "role") == "alert"
    page.close()


def test_desktop_side_panel_still_logs_errors(browser, server) -> None:
    """AC-8.2 — พฤติกรรมเดิมของแผงข้างต้องไม่หาย"""
    page = open_page(browser, server)
    page.evaluate("handleJson({type:'log', level:'error', text:'พังจริง'})")
    assert page.is_visible(".side")
    assert "พังจริง" in page.inner_text("#logs")
    assert page.is_visible("#alert")
    assert page.locator("#alert .msg").inner_text().count("พังจริง") == 1
    page.close()


def test_error_clears_when_things_work_again(browser, server) -> None:
    """AC-8.3"""
    page = open_page(browser, server, width=390, height=844)
    page.evaluate("handleJson({type:'log', level:'error', text:'เน็ตหลุด'})")
    assert page.is_visible("#alert")
    page.evaluate("handleJson({type:'speech', state:'start'})")
    assert page.is_hidden("#alert"), "error ค้างอยู่ทั้งที่ระบบกลับมาปกติ"
    assert orb_state(page) == "listening"
    page.close()


# ─────────────────────────────────────────────────── P2-9 เชื่อมต่อใหม่เอง
def test_client_reconnects_after_a_server_restart(browser, server) -> None:
    """AC-9.1 / AC-9.2 — ต่อกลับเองภายใน 15 วินาที + มีเส้นคั่นบอกว่าความจำเริ่มใหม่"""
    page = open_page(browser, server)
    page.evaluate("connect()")
    page.wait_for_function("window.__vl && sessions === 1", timeout=5000)
    page.evaluate("beginMsg('user'); writeMsg('ข้อความก่อนหลุด'); endMsg()")

    server.stop()
    page.wait_for_function("live === false", timeout=5000)
    assert "กำลังเชื่อมต่อใหม่" in page.inner_text("#status-now")

    server.start()
    page.wait_for_function("sessions === 2", timeout=20000)
    assert page.evaluate("window.__vl.reconnects") >= 1
    marks = page.locator("#stream .reset-mark")
    assert marks.count() == 1, "ไม่มีเส้นคั่นบอกว่าความจำ AI เริ่มใหม่"
    assert "AI ไม่จำข้อความด้านบน" in marks.first.inner_text()
    assert "ข้อความก่อนหลุด" in page.inner_text("#stream")
    page.close()


def test_retries_are_capped_then_a_button_appears(browser, server) -> None:
    """AC-9.3 / AC-9.4 — ไม่ retry ถี่ไม่จำกัด และหน้าไม่พังระหว่างนั้น"""
    page = open_page(browser, server)
    page.evaluate("RETRY_DELAYS.splice(0, RETRY_DELAYS.length, 60, 60, 60)")
    port = free_port()                    # พอร์ตที่ไม่มีใครฟัง
    page.evaluate(f"""() => {{
        R.wanted = true;
        window.__origWS = WebSocket;
        window.WebSocket = function () {{ return new window.__origWS('ws://127.0.0.1:{port}/ws'); }};
    }}""")
    page.evaluate("connect()")
    page.wait_for_function("window.__vl.attempts >= 4", timeout=8000)
    page.wait_for_timeout(600)
    attempts = page.evaluate("window.__vl.attempts")
    assert attempts == 4, f"ลองใหม่เกินเพดาน ({attempts} ครั้ง)"
    assert page.is_enabled("#btn-start")
    assert "เชื่อมต่อใหม่" in page.inner_text("#btn-start")
    assert page.is_visible("#alert")
    # ผู้ใช้พูดระหว่างนั้นต้องไม่ทำให้หน้าพัง (เสียงถูกทิ้งเงียบ ๆ)
    page.evaluate("send({type:'text', text:'ทดสอบระหว่างหลุด'})")
    assert page.evaluate("typeof handleJson === 'function'")
    page.close()


# ────────────────────────────────────────────────────────── P2-10 scroll
FILLER = ("for (let i = 0; i < 40; i++) "
          "{ beginMsg('assistant'); writeMsg('ข้อความยาวพอให้เกิดแถบเลื่อน '.repeat(4)); endMsg(); }")


def test_scroll_sticks_to_bottom_but_not_while_reading_back(browser, server) -> None:
    """AC-10.1 / AC-10.2 / AC-10.3 / AC-10.4"""
    page = open_page(browser, server, width=1440, height=700)
    page.evaluate(FILLER)
    page.wait_for_timeout(100)
    at_bottom = page.evaluate(
        "() => { const s = $('stream'); return s.scrollHeight - s.scrollTop - s.clientHeight; }")
    assert at_bottom <= 2, "เปิดมาแล้วควรอยู่ก้น stream"
    assert page.is_hidden("#jump")

    # เลื่อนขึ้นไปอ่านย้อน แล้วให้ข้อความใหม่เข้ามา
    page.evaluate("$('stream').scrollTop = 0")
    page.wait_for_timeout(120)
    page.evaluate("beginMsg('assistant'); writeMsg('ข้อความใหม่ระหว่างอ่านย้อน'); endMsg()")
    page.wait_for_timeout(120)
    assert page.evaluate("$('stream').scrollTop") == 0, "จอกระชากลงล่างทั้งที่กำลังอ่านย้อน"
    page.locator("#jump").wait_for(state="visible", timeout=2000)

    page.click("#jump")
    page.wait_for_timeout(150)
    assert page.is_hidden("#jump")
    left = page.evaluate(
        "() => { const s = $('stream'); return s.scrollHeight - s.scrollTop - s.clientHeight; }")
    assert left <= 2

    # กลับมาอยู่ก้นเองแล้วต้อง auto-scroll ต่อ
    page.evaluate("beginMsg('assistant'); writeMsg('ตามต่ออัตโนมัติ'); endMsg()")
    page.wait_for_timeout(120)
    left = page.evaluate(
        "() => { const s = $('stream'); return s.scrollHeight - s.scrollTop - s.clientHeight; }")
    assert left <= 2
    page.close()


# ──────────────────────────────────────────────────── P2-13 ปุ่มโหมดเสียง
def test_mode_button_label_matches_server_config(browser, server) -> None:
    """AC-13.1 / AC-13.2 / AC-13.3 / AC-13.4"""
    page = open_page(browser, server)
    page.evaluate("connect()")
    page.wait_for_function("sessions === 1", timeout=5000)

    btn = page.locator("#btn-echo")
    assert "โหมดลำโพง" in btn.inner_text(), "ป้ายไม่ตรงกับ echo_guard=true ที่เซิร์ฟเวอร์ส่งมา"
    assert btn.get_attribute("aria-pressed") == "true"

    btn.click()
    assert "โหมดหูฟัง" in btn.inner_text()
    assert btn.get_attribute("aria-pressed") == "false"
    assert btn.get_attribute("data-on") == "0"

    page.evaluate("handleJson({type:'setting', key:'echo_guard', on:true})")
    assert "โหมดลำโพง" in btn.inner_text(), "event setting ไม่ทำให้ป้ายตรงกับค่าจริง"
    assert btn.get_attribute("aria-pressed") == "true"

    mute = page.locator("#btn-mute")
    page.evaluate("handleJson({type:'setting', key:'mute', on:true})")
    assert "เสียงปิด" in mute.inner_text()
    assert mute.get_attribute("aria-pressed") == "true"
    page.close()


# ─────────────────────────────────────────── P2-14 error ของไมโครโฟน
NO_SECURE = """
Object.defineProperty(window, 'isSecureContext', { value: false });
Object.defineProperty(navigator, 'mediaDevices', { value: undefined });
"""


def test_insecure_origin_is_reported_before_any_click(browser, server) -> None:
    """AC-14.1 — เปิดผ่าน http://<ip> ต้องรู้ทันทีตอนโหลดหน้า"""
    page = open_page(browser, server, width=390, height=844, init=NO_SECURE)
    alert = page.locator("#alert")
    alert.wait_for(state="visible", timeout=2000)
    text = alert.inner_text()
    assert "https://" in text and "localhost" in text
    assert page.is_disabled("#btn-start")
    assert "https" in (page.get_attribute("#btn-start", "title") or "")
    page.close()


def _gum_fails(name: str) -> str:
    return """
Object.defineProperty(navigator, 'mediaDevices', {
  value: { getUserMedia: () => Promise.reject(new DOMException('จำลอง', '%s')) },
});
""" % name


@pytest.mark.parametrize(("name", "needle"), [
    ("NotAllowedError", "ปฏิเสธสิทธิ์"),
    ("NotFoundError", "ไม่พบไมโครโฟน"),
    ("NotReadableError", "แอปอื่นใช้อยู่"),
    ("WeirdBrandNewError", "WeirdBrandNewError"),
])
def test_each_mic_failure_gets_its_own_message(browser, server, name, needle) -> None:
    """AC-14.2 / AC-14.3 / AC-14.4"""
    page = open_page(browser, server, init=_gum_fails(name))
    page.click("#btn-start")
    page.locator("#alert").wait_for(state="visible", timeout=3000)
    text = page.locator("#alert .msg").inner_text()
    assert needle in text, f"{name} ได้ข้อความผิด: {text!r}"
    assert page.is_enabled("#btn-start"), "ต้องกดลองใหม่ได้"
    page.close()


# ──────────────────────────────────────────────────────── P3-16 a11y (โมดัลคู่มือ)
def test_guide_modal_traps_focus_and_gives_it_back(browser, server) -> None:
    """AC-16.3 — Tab วนอยู่ในโมดัล กด Esc ปิดได้ และโฟกัสกลับไปที่ปุ่มที่เปิด"""
    page = open_page(browser, server)
    page.focus("#btn-help")
    page.click("#btn-help")
    page.wait_for_function("$('overlay').open === true")

    # ไล่ Tab หลายรอบ โฟกัสต้องอยู่ในโมดัลตลอด ไม่หลุดไปปุ่มด้านหลัง
    seen = []
    for _ in range(12):
        page.keyboard.press("Tab")
        seen.append(page.evaluate("""() => {
          const el = document.activeElement;
          if (!el || el === document.body) return '(นอกหน้าเว็บ)';
          return ($('overlay').contains(el) ? 'ใน:' : 'นอก:') + (el.id || el.tagName);
        }"""))
    # Chrome พา Tab ออกไปที่แถบเครื่องมือของเบราว์เซอร์ก่อนวนกลับ (พฤติกรรมปกติของ
    # modal dialog) แต่ต้องไม่ไปหยุดที่ปุ่มของหน้าเว็บที่อยู่ข้างหลังโมดัลเด็ดขาด
    outside = [s for s in seen if s.startswith('นอก:')]
    assert outside == [], f"โฟกัสไปโดนของที่อยู่หลังโมดัล: {outside} (ทั้งหมด {seen})"
    assert any(s.startswith('ใน:') for s in seen)

    page.keyboard.press("Escape")
    page.wait_for_function("$('overlay').open === false")
    assert page.evaluate("() => document.activeElement.id") == "btn-help", (
        "ปิดโมดัลแล้วโฟกัสต้องกลับไปที่ปุ่มที่เปิดมัน")
    page.close()


def test_guide_modal_hides_the_page_behind_it(browser, server) -> None:
    """AC-16.3 — background ต้องเป็น inert (คลิก/โฟกัสไม่โดน)"""
    page = open_page(browser, server)
    page.click("#btn-help")
    page.wait_for_function("$('overlay').open === true")
    page.evaluate("() => $('btn-start').focus()")
    assert page.evaluate("() => document.activeElement.id") != "btn-start"
    page.close()


def test_focus_ring_is_actually_painted_on_both_themes(browser, server) -> None:
    """AC-16.2 — เดิมวงโฟกัสเป็น cyan จาง 8% (1.14:1) คือมองไม่เห็น"""
    page = open_page(browser, server)
    for theme in ("hud", "clay"):
        page.evaluate("(t) => applyTheme(t)", theme)
        page.keyboard.press("Tab")               # เข้าโหมดคีย์บอร์ด (:focus-visible)
        page.focus("#btn-start")
        shadow = page.evaluate(
            "() => getComputedStyle($('btn-start')).boxShadow")
        assert shadow and shadow != "none", f"ธีม {theme}: ไม่มีวงโฟกัสเลย"
        assert shadow.count("rgb") >= 2, f"ธีม {theme}: วงโฟกัสไม่ครบสองชั้น ({shadow})"
    page.close()


def test_finished_reply_is_announced_exactly_once(browser, server) -> None:
    """AC-16.1 — ประกาศ 1 ครั้งตอนจบ ไม่ใช่ทุก token"""
    page = open_page(browser, server)
    page.evaluate("""() => {
      handleJson({ type: 'begin', role: 'assistant' });
      for (const t of ['สวัสดี', 'ครับ', ' ยินดี', 'ต้อนรับ']) {
        handleJson({ type: 'delta', text: t });
      }
    }""")
    assert page.locator("#sr-live").inner_text().strip() == ""
    page.evaluate("() => handleJson({ type: 'end' })")
    page.wait_for_function("$('sr-live').textContent.trim().length > 0")
    said = page.locator("#sr-live").inner_text()
    assert "สวัสดีครับ ยินดีต้อนรับ" in said
    assert said.count("สวัสดี") == 1
    page.close()


def test_whole_flow_works_with_the_keyboard_only(browser, server) -> None:
    """AC-16.5 — เริ่ม → หยุด → เปิดคู่มือ → ปิด ด้วยคีย์บอร์ดล้วน"""
    page = open_page(browser, server)
    page.focus("#btn-start")
    page.keyboard.press("Enter")
    # เผื่อเวลาให้ WebSocket + AudioContext ตั้งตัว (เครื่อง CI ช้ากว่านี้ได้อีก)
    page.wait_for_function("live === true", timeout=20000)
    page.focus("#btn-stop")
    page.keyboard.press("Enter")
    page.focus("#btn-help")
    page.keyboard.press("Enter")
    page.wait_for_function("$('overlay').open === true")
    page.keyboard.press("Escape")
    page.wait_for_function("$('overlay').open === false")
    page.close()
