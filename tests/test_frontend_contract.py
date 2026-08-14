"""P2-7 … P2-14 — สัญญาระดับซอร์สของฝั่งเบราว์เซอร์

เทสต์ชุดนี้ไม่ได้แทนการทดสอบพฤติกรรมจริง (อยู่ที่ tests/test_frontend_ui.py ซึ่ง
ใช้ Chromium จริง แต่ต้องสั่งด้วย `pytest -m browser`) — ที่นี่กันการถอยหลังของ
"สายไฟ" ที่ขาดมาก่อน เช่น markup ที่มีแต่ไม่มีโค้ดขับ หรือค่า fallback ที่ผิด
เพราะบั๊กเหล่านั้นกลับมาได้เงียบ ๆ และชุดเทสต์เบราว์เซอร์ไม่ได้รันทุกครั้ง
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
JS = (STATIC / "app.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────── P2-7 วงแหวนรอบวงกลมไมค์
def test_orb_ring_is_actually_driven() -> None:
    """AC-7.1 — เดิม grep 'orb-val' ในโค้ดได้ 0 ผลลัพธ์ (markup ตายสนิท)"""
    assert "orb-val" in JS, "#orb-val ยังไม่มีโค้ดขับเลย"
    assert "stroke-dashoffset" in JS
    assert "requestAnimationFrame" in JS.split("function startOrbLoop")[1][:600], (
        "ควรหน่วงค่าด้วย requestAnimationFrame ไม่เขียนทุก event ดิบ ๆ")
    assert re.search(r"case 'level'[\s\S]{0,700}orbLevel\(", JS), (
        "event level ยังไม่ถูกต่อเข้ากับวงแหวน")


def test_search_icon_maps_to_its_own_state() -> None:
    """AC-7.2 — ระหว่างค้นเน็ตต้องเป็น searching (สีม่วงที่ประกาศไว้แต่ไม่เคยใช้)"""
    table = JS[JS.index("const ORB_BY_ICON"):JS.index("const ORB_BY_PHASE")]
    assert "'🔎': 'searching'" in table
    assert '.orb[data-state="searching"]' in CSS
    assert "--violet" in CSS[CSS.index('.orb[data-state="searching"]'):][:400]


def test_unknown_icon_falls_back_to_thinking() -> None:
    """AC-7.3 — fallback ต้องไม่ใช่ 'idle' (บอกว่าพร้อมฟังทั้งที่ยังยุ่ง)"""
    assert re.search(r"ORB_BY_ICON\[m\.icon\] \|\| 'thinking'", JS)
    assert "ORB_BY_ICON[m.icon] || 'idle'" not in JS


def test_thinking_state_keeps_moving_and_respects_reduced_motion() -> None:
    """AC-7.4 / AC-7.5"""
    assert "@keyframes orb-sweep" in CSS
    assert re.search(r'\.orb\[data-state="thinking"\] svg\.ring[\s\S]{0,120}animation', CSS)
    reduced = CSS[CSS.index("@media (prefers-reduced-motion: reduce)"):]
    assert "animation: none" in reduced
    assert "prefers-reduced-motion" in JS, "ฝั่ง JS ต้องเคารพค่านี้ด้วย"


# ───────────────────────────────────────────── P2-8 error ที่เห็นได้บนมือถือ
def test_error_surface_lives_in_the_dock_not_the_side_panel() -> None:
    """AC-8.1 — .side ถูก display:none ที่ ≤1000px แผง error จึงต้องไม่อยู่ในนั้น"""
    dock = HTML[HTML.index('<footer class="dock">'):HTML.index("</footer>")]
    assert 'id="alert"' in dock, "แผงแจ้งเตือนต้องอยู่ใน dock"
    aside = HTML[HTML.index('<aside class="side">'):HTML.index("</aside>")]
    assert 'id="alert"' not in aside
    assert ".dock-alert" in CSS
    small = CSS[CSS.index("@media (max-width: 1000px)"):]
    assert ".dock-alert" not in small.split("}")[0], "ห้ามซ่อนแผงนี้บนจอเล็ก"


def test_backend_error_events_reach_the_dock_alert() -> None:
    block = JS[JS.index("case 'log':"):JS.index("case 'closed':")]
    assert "showAlert(" in block and "'error'" in block


def test_error_state_is_cleared_automatically() -> None:
    """AC-8.3 — error ต้องไม่ค้างเมื่อระบบกลับมาปกติ"""
    for anchor in ("case 'ready':", "case 'speech':", "case 'begin':"):
        block = JS[JS.index(anchor):JS.index(anchor) + 700]
        assert "clearAlert(" in block, f"ไม่ได้เคลียร์ error ที่ {anchor}"


def test_error_orb_state_exists_and_is_not_colour_only() -> None:
    """AC-8.4"""
    assert '.orb[data-state="error"]' in CSS
    assert "--red" in CSS[CSS.index('.orb[data-state="error"]'):][:300]
    assert 'role="alert"' in HTML, "ต้องมีข้อความประกอบ ไม่พึ่งสีอย่างเดียว"


# ─────────────────────────────────────────────────── P2-9 เชื่อมต่อใหม่เอง
def test_auto_reconnect_uses_capped_backoff() -> None:
    """AC-9.1 / AC-9.3"""
    delays = re.search(r"const RETRY_DELAYS = \[([\d,\s]+)\]", JS)
    assert delays, "ไม่พบตารางเวลาการลองใหม่"
    values = [int(v) for v in delays.group(1).split(",") if v.strip()]
    assert values == sorted(values) and values[-1] <= 15000, values
    assert len(values) >= 4
    assert "R.tries >= RETRY_DELAYS.length" in JS, "ต้องมีเพดานจำนวนครั้ง"
    # H-01: token เดินทางมากับ cookie ที่เซิร์ฟเวอร์ตั้งไว้ ไม่ใช่ query string
    assert "token=" not in JS[JS.index("function connect()"):][:800], (
        "token ต้องไม่อยู่ใน URL ของ WebSocket — รั่วเข้า log ของ proxy/CDN (H-01)")
    assert "/ws?token=" not in JS


def test_new_session_marks_that_memory_restarted() -> None:
    """AC-9.2"""
    assert "function markMemoryReset" in JS
    assert "AI ไม่จำข้อความด้านบน" in JS
    assert ".reset-mark" in CSS
    ready = JS[JS.index("case 'ready':"):JS.index("case 'phase':")]
    assert "markMemoryReset()" in ready and "sessions > 1" in ready


def test_reconnecting_state_is_visible_to_the_user() -> None:
    assert "กำลังเชื่อมต่อใหม่..." in JS


# ──────────────────────────────────────────────────────── P2-10 การเลื่อนจอ
def test_scroll_is_sticky_bottom_not_forced() -> None:
    """AC-10.1 / AC-10.2"""
    body = JS[JS.index("function scrollStream"):JS.index("function jumpToLatest")]
    assert "stickBottom" in body, "ยัง force เลื่อนลงก้นทุก token"
    assert "const STICK_PX = 48" in JS
    assert "function nearBottom" in JS
    assert re.search(r"\$\('stream'\)\.addEventListener\('scroll'", JS)


def test_jump_to_latest_button_exists() -> None:
    """AC-10.3 / AC-10.4"""
    assert 'id="jump"' in HTML and "↓ ล่าสุด" in HTML
    assert "function jumpToLatest" in JS
    assert re.search(r"\$\('jump'\)\.addEventListener\('click', jumpToLatest\)", JS)
    jump = CSS[CSS.index(".jump {"):CSS.index(".jump[hidden]")]
    assert "min-height: 44px" in jump, "เป้าแตะต้องไม่เล็กกว่า 44px"


# ──────────────────────────────────────────────── P2-13 ป้ายปุ่มโหมดเสียง
def test_mode_buttons_render_from_one_place() -> None:
    """AC-13.2 — textContent / dataset.on / aria-pressed ต้องเปลี่ยนจากจุดเดียว"""
    for fn in ("function renderEcho", "function renderMute"):
        body = JS[JS.index(fn):JS.index(fn) + 700]
        assert "dataset.on" in body
        assert "aria-pressed" in body
        assert "textContent" in body


def test_headphone_button_label_says_the_mode_in_use() -> None:
    """AC-13.1 — ป้ายต้องบอกโหมดที่กำลังใช้อยู่ ไม่ใช่กลับด้าน"""
    body = JS[JS.index("function renderEcho"):JS.index("function renderMute")]
    assert "guardOn ? '🔈 โหมดลำโพง' : '🎧 โหมดหูฟัง'" in body
    ready = JS[JS.index("case 'ready':"):JS.index("case 'phase':")]
    assert "renderEcho(" in ready and "renderMute(" in ready


def test_setting_event_keeps_the_label_in_sync() -> None:
    """AC-13.3 — เดิม event setting แก้แค่ dataset.on ป้ายจึง desync"""
    block = JS[JS.index("case 'setting':"):JS.index("case 'cleared':")]
    assert "renderMute(" in block and "renderEcho(" in block
    assert "dataset.on" not in block, "ยังแก้ dataset ตรง ๆ นอกฟังก์ชันวาดปุ่ม"


def test_buttons_declare_aria_pressed_in_markup() -> None:
    """AC-13.4"""
    for btn in ("btn-mute", "btn-echo"):
        tag = re.search(r'<button[^>]*id="%s"[^>]*>' % btn, HTML)
        assert tag and "aria-pressed" in tag.group(0), btn


# ───────────────────────────────── P2-14 ข้อความ error ของไมค์ที่ตรงสาเหตุ
def test_secure_context_is_checked_before_any_click() -> None:
    """AC-14.1"""
    assert "window.isSecureContext" in JS
    assert "function checkAudioSupport" in JS
    init = JS[JS.index("(async function init()"):]
    assert "checkAudioSupport()" in init, "ต้องเช็คตอนโหลดหน้า ไม่ใช่ตอนกดปุ่ม"
    body = JS[JS.index("function checkAudioSupport"):JS.index("async function start")]
    assert "btn.disabled = true" in body, "ต้อง disable ปุ่มเริ่มพร้อมคำอธิบาย"
    assert "https://" in JS and "http://localhost" in JS


@pytest.mark.parametrize("name", ["NotAllowedError", "NotFoundError",
                                  "NotReadableError", "SecurityError"])
def test_each_mic_error_has_its_own_message(name) -> None:
    """AC-14.2 / AC-14.3"""
    body = JS[JS.index("function micErrorText"):JS.index("function checkAudioSupport")]
    assert f"case '{name}'" in body


def test_unknown_mic_error_shows_the_real_name() -> None:
    """AC-14.4 — support ต้องวินิจฉัยได้จากข้อความที่ผู้ใช้เห็น"""
    body = JS[JS.index("function micErrorText"):JS.index("function checkAudioSupport")]
    default = body[body.index("default:"):]
    assert "name" in default and "err.message" in default
    assert "ตรวจสิทธิ์ไมค์ของเบราว์เซอร์แล้วลองใหม่" not in JS, (
        "ยังมีข้อความเดิมที่ใช้ตอบทุกสาเหตุ")
