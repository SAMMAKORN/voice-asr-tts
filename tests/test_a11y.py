"""P3-16 / P3-17 — การเข้าถึง (a11y) และค่าคอนทราสต์ของทั้งสองธีม

ตรวจที่ระดับซอร์สเหมือน `test_frontend_contract.py` เพราะสิ่งที่ต้องกันคือการ
"ถอยกลับเงียบ ๆ" ของ token สีและ attribute ที่มองด้วยตาไม่เห็นความต่าง
ส่วนพฤติกรรมจริง (Tab วนในโมดัล, VoiceOver อ่านกี่ครั้ง) อยู่ใน
`tests/test_frontend_ui.py` ซึ่งต้องสั่งด้วย `pytest -m browser`

สูตรคอนทราสต์ตาม WCAG 2.1 (relative luminance) คำนวณเองในไฟล์นี้ ไม่พึ่ง
ไลบรารีภายนอก เพื่อไม่ให้ต้องเพิ่ม dependency
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


# ──────────────────────────────────────────────────────── คำนวณคอนทราสต์ WCAG
def _srgb(channel: int) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb(r) + 0.7152 * _srgb(g) + 0.0722 * _srgb(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def theme_tokens(selector: str) -> dict[str, str]:
    """ดึงคู่ `--token: ค่า;` ของบล็อกธีมหนึ่งบล็อกออกมาเป็น dict"""
    start = CSS.index(selector)
    body = CSS[CSS.index("{", start) + 1:]
    depth, out = 1, []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    block = "".join(out)
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", block)}


HUD = theme_tokens(":root {")
CLAY = theme_tokens(':root[data-theme="clay"] {')


def test_contrast_helper_matches_known_values() -> None:
    """กันสูตรเพี้ยนเอง — ขาวบนดำต้องได้ 21:1 พอดี"""
    assert round(contrast("#ffffff", "#000000"), 1) == 21.0
    assert round(contrast("#777777", "#ffffff"), 2) == 4.48


# ────────────────────────────────────────────── P3-16 ประกาศสถานะให้ screen reader
def test_status_area_is_a_polite_live_region() -> None:
    """AC-16.4 — แถบสถานะต้องประกาศเองเมื่อเปลี่ยน"""
    block = HTML[HTML.index('class="status-text"'):][:200]
    assert 'role="status"' in block
    assert 'aria-live="polite"' in block


def test_log_panel_is_a_log_live_region() -> None:
    """AC-16.4 — แผงบันทึกระบบเป็น role=log"""
    block = HTML[HTML.index('id="logs"'):][:200]
    assert 'role="log"' in block
    assert 'aria-live="polite"' in block


def test_stream_is_not_a_live_region() -> None:
    """AC-16.1 — ห้ามประกาศที่ #stream โดยตรง (ต่อทีละ token = อ่านซ้ำทั้งย่อหน้า)"""
    block = HTML[HTML.index('class="stream" id="stream"'):][:160]
    assert 'aria-live="off"' in block
    assert 'aria-live="polite"' not in block
    assert 'role="log"' not in block


def test_finished_message_is_announced_once_from_a_separate_element() -> None:
    """AC-16.1 — ประกาศครั้งเดียวตอนจบข้อความ ผ่าน element ที่ซ่อนไว้"""
    assert 'id="sr-live"' in HTML
    assert 'aria-live="polite"' in HTML[HTML.index('id="sr-live"') - 120:
                                        HTML.index('id="sr-live"') + 160]
    assert "function announce(" in JS
    end_body = JS.split("function endMsg(")[1].split("\n}")[0]
    assert "announce(" in end_body, "endMsg() ต้องเป็นคนประกาศ ไม่ใช่ writeMsg()"
    write_body = JS.split("function writeMsg(")[1].split("\n}")[0]
    assert "announce(" not in write_body, "ห้ามประกาศระหว่างสตรีมทีละ token"


# ─────────────────────────────────────────────────────── P3-16 วงโฟกัสคีย์บอร์ด
@pytest.mark.parametrize("theme,tokens", [("hud", HUD), ("clay", CLAY)])
def test_focus_ring_contrast_passes_on_both_themes(theme: str, tokens: dict) -> None:
    """AC-16.2 — เดิม cyan จาง 8% ได้ 1.14:1 คือมองไม่เห็นเลย"""
    focus = tokens["--focus"]
    assert "var(--cyan)" in focus and "var(--bg)" in focus, (
        f"ธีม {theme}: วงโฟกัสต้องเป็นวงทึบสองชั้น ไม่ใช่สีจาง — ได้ {focus!r}")
    assert "color-mix" not in focus, f"ธีม {theme}: ยังใช้สีจางอยู่"
    ratio = contrast(tokens["--cyan"], tokens["--bg"])
    assert ratio >= 3.0, f"ธีม {theme}: วงโฟกัสได้ {ratio:.2f}:1 (ต้อง ≥ 3:1)"


def test_buttons_declare_a_focus_visible_style() -> None:
    """AC-16.2 — ปุ่มทุกตัวใช้คลาส .btn ร่วมกัน จึงต้องมีกฎ :focus-visible ของคลาสนี้"""
    assert ".btn:focus-visible" in CSS
    assert ".jump:focus-visible" in CSS
    assert ".compose input:focus-visible" in CSS
    rule = CSS.split(".btn:focus-visible")[1][:400]
    assert "var(--focus)" in rule


def test_every_button_in_markup_is_a_real_button() -> None:
    """AC-16.5 — ใช้คีย์บอร์ดอย่างเดียวได้ ต้องไม่มี div ที่ทำตัวเป็นปุ่ม"""
    fake = re.findall(r'<div[^>]*class="[^"]*\bbtn\b[^"]*"', HTML)
    assert fake == [], f"พบ div ที่ทำตัวเป็นปุ่ม: {fake}"


# ──────────────────────────────────────────────────────── P3-16 โมดัลคู่มือ
def test_guide_modal_is_a_native_dialog() -> None:
    """AC-16.3 — <dialog> ให้ focus trap / Esc / คืนโฟกัส มาจากเบราว์เซอร์เอง"""
    assert re.search(r'<dialog[^>]*id="overlay"', HTML), "โมดัลคู่มือยังไม่ใช่ <dialog>"
    assert "showModal()" in JS
    assert re.search(r"function closeTour\(\)[\s\S]{0,240}\.close\(\)", JS)
    assert 'aria-labelledby="tour-label"' in HTML
    # ห้ามเหลือทางเปิดโมดัลแบบ div เก่าที่ไม่มี focus trap
    assert "$('overlay').hidden = false" not in JS


def test_guide_modal_remembers_it_was_seen_on_any_close_path() -> None:
    """ปิดด้วย Esc ก็ต้องจำ — ไม่งั้นคู่มือเด้งซ้ำทุกครั้งที่รีเฟรช"""
    assert re.search(r"addEventListener\('close', markTourSeen\)", JS)


def test_dialog_is_hidden_when_not_open() -> None:
    """`display: grid` ของ .overlay ทับกฎ UA — ต้องประกาศเองไม่งั้นค้างเต็มจอ"""
    assert ".overlay:not([open])" in CSS


# ────────────────────────────────────────────── P3-17 คอนทราสต์ของธีมดินน้ำมัน
CLAY_SURFACES = ("--panel-solid", "--bg", "--bg-2")


@pytest.mark.parametrize("surface", CLAY_SURFACES)
def test_clay_faint_text_passes_wcag_aa(surface: str) -> None:
    """AC-17.1 — เดิม #8c7c88 ได้ 3.33-3.78:1 ต่ำกว่าเกณฑ์ 4.5:1 ทุกพื้น"""
    ratio = contrast(CLAY["--faint"], CLAY[surface])
    assert ratio >= 4.5, f"--faint บน {surface} ได้ {ratio:.2f}:1"


def test_clay_faint_is_still_fainter_than_body_text() -> None:
    """AC-17.3 — ลำดับชั้นสายตาต้องอยู่: text เข้มสุด → dim → faint"""
    bg = CLAY["--panel-solid"]
    text = contrast(CLAY["--text"], bg)
    dim = contrast(CLAY["--dim"], bg)
    faint = contrast(CLAY["--faint"], bg)
    assert text > dim > faint


def test_hud_theme_colours_are_untouched() -> None:
    """AC-17.2 — ห้ามแตะสีของธีม hud (ผ่านเกณฑ์อยู่แล้ว)"""
    assert HUD["--faint"] == "#8199ae"
    assert HUD["--text"] == "#e3eef8"
    assert HUD["--dim"] == "#a3b8cb"
