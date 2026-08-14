"""P3-19 — ตรวจค่าคอนฟิกตอนเริ่มโปรแกรม

เดิมค่าที่ตั้งผิดถูกกลืนเงียบ (`_f()` คืนค่าปริยายเมื่อ ValueError) ผู้ใช้จึงจูนค่า
ไปแล้วไม่มีผลโดยไม่รู้ตัว และคู่ค่าที่รวมกันแล้วทำให้รอเงียบ 16 วินาที
(TTS_SINGLE_REQUEST=1 + ข้อความยาว) ก็ไม่มีอะไรเตือน
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vc import config as config_mod
from vc.config import DEFAULT_TZ, RANGES, Config, check_combinations, load_config

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
BOOL_KEYS = ("ECHO_GUARD", "TTS_SINGLE_REQUEST", "TTS_SEARCH_FILLER",
             "WEB_SEARCH", "LOG_TRANSCRIPT")


@pytest.fixture
def clean_env(monkeypatch):
    """สภาพแวดล้อมสะอาด: ไม่อ่าน .env ของเครื่อง และไม่มีค่าเก่าค้าง"""
    monkeypatch.setattr(config_mod, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config_mod, "_warned", set())
    for key in (*RANGES, *BOOL_KEYS, "LOG_DIR", "LANG_HINT", "SYSTEM_PROMPT",
                "CHAT_MODEL", "ASR_MODEL", "TTS_MODEL", "APP_TZ"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("API_KEY", "test-key")
    return monkeypatch


# ────────────────────────────────────────────── ค่าผิดชนิด (AC-19.1)
def test_non_numeric_value_names_the_key_and_the_accepted_range(clean_env) -> None:
    """AC-19.1 — ต้องไม่ใช่ `ValueError: invalid literal for int()`"""
    clean_env.setenv("VAD_ABS_THRESHOLD", "abc")
    with pytest.raises(SystemExit) as err:
        load_config()
    msg = str(err.value)
    assert "VAD_ABS_THRESHOLD" in msg
    assert "'abc'" in msg, "ต้องบอกค่าที่ได้รับมาด้วย"
    assert "0.0001" in msg and "1" in msg, "ต้องบอกช่วงที่ยอมรับ"
    assert err.value.code != 0


@pytest.mark.parametrize("bad", ["", "  ", "nan", "inf", "1,5", "12ก"])
def test_every_shape_of_garbage_is_caught(clean_env, bad: str) -> None:
    clean_env.setenv("SEARCH_TIMEOUT", bad)
    if not bad.strip():
        assert load_config().search_timeout == 12.0   # ว่าง = ใช้ค่าปริยาย
        return
    with pytest.raises(SystemExit) as err:
        load_config()
    assert "SEARCH_TIMEOUT" in str(err.value)


def test_boolean_keys_reject_nonsense_without_crashing(clean_env) -> None:
    clean_env.setenv("WEB_SEARCH", "maybe")
    cfg = load_config()
    assert cfg.web_search is True, "ค่าที่อ่านไม่ออกต้องกลับไปใช้ค่าปริยาย"
    assert any("WEB_SEARCH" in w for w in cfg.warnings)


def test_unknown_timezone_is_normalized_to_the_real_fallback(clean_env, capsys) -> None:
    clean_env.setenv("APP_TZ", "Invalid/Config_Zone")
    cfg = load_config()

    assert cfg.tz == DEFAULT_TZ
    assert "APP_TZ" in capsys.readouterr().err


# ────────────────────────────────────────────── ค่านอกช่วง (AC-19.3)
def test_out_of_range_value_is_clamped_with_a_warning(clean_env) -> None:
    """AC-19.3 — เลือกทาง clamp + WARNING (บันทึกไว้ใน README/.env.example)"""
    clean_env.setenv("CHAT_TEMPERATURE", "5.0")
    cfg = load_config()
    assert cfg.temperature == 2.0
    assert any("CHAT_TEMPERATURE" in w and "5.0" in w for w in cfg.warnings)


@pytest.mark.parametrize(("key", "raw", "attr", "want"), [
    ("SEARCH_RESULTS", "500", "search_results", 20),
    ("TTS_MAX_CHARS", "1", "tts_max_chars", 40),
    ("VAD_NOISE_MULT", "-3", "vad_noise_mult", 1.0),
    ("HTTP_RETRY_MAX", "99", "http_retry_max", 10),
    ("LOG_RETENTION_DAYS", "-1", "log_retention_days", 0),
])
def test_clamping_uses_the_documented_bounds(clean_env, key, raw, attr, want) -> None:
    clean_env.setenv(key, raw)
    cfg = load_config()
    assert getattr(cfg, attr) == want
    assert any(key in w for w in cfg.warnings)


def test_values_inside_the_range_are_left_alone(clean_env) -> None:
    clean_env.setenv("VAD_ABS_THRESHOLD", "0.02")
    clean_env.setenv("TTS_MAX_CHARS", "180")
    cfg = load_config()
    assert cfg.vad_abs_threshold == 0.02 and cfg.tts_max_chars == 180
    assert cfg.warnings == []


# ────────────────────────────────────── คู่ค่าที่รวมกันแล้วเป็นปัญหา (AC-19.2)
def test_single_request_with_a_long_cap_warns_about_the_wait(clean_env) -> None:
    """AC-19.2 — เคสจริงจากบันทึก: 608 ตัวอักษรก้อนเดียว = รอเงียบ 16 วินาที"""
    clean_env.setenv("TTS_SINGLE_REQUEST", "1")
    clean_env.setenv("TTS_MAX_CHARS", "600")
    cfg = load_config()
    hit = [w for w in cfg.warnings if "TTS_SINGLE_REQUEST" in w]
    assert hit, "ต้องมี WARNING เรื่องนี้"
    assert re.search(r"\d+ วินาที", hit[0]), "ต้องบอกเวลาที่คาดว่าจะรอ"
    assert cfg.tts_max_chars == 600, "ระบบต้องยังรันได้ (WARNING ไม่ใช่ error)"


def test_streaming_tts_with_the_same_cap_is_fine(clean_env) -> None:
    clean_env.setenv("TTS_SINGLE_REQUEST", "0")
    clean_env.setenv("TTS_MAX_CHARS", "600")
    assert load_config().warnings == []


def test_first_chunk_larger_than_the_next_one_warns() -> None:
    cfg = Config(tts_first_chars=200, tts_chunk_chars=60)
    assert any("TTS_FIRST_CHARS" in w for w in check_combinations(cfg))


def test_chunk_larger_than_the_per_request_cap_warns() -> None:
    cfg = Config(tts_chunk_chars=500, tts_max_chars=260)
    assert any("TTS_CHUNK_CHARS" in w for w in check_combinations(cfg))


def test_impossible_utterance_window_warns() -> None:
    cfg = Config(vad_min_utterance_ms=9000, vad_max_utterance_ms=2000)
    assert any("VAD_MIN_UTTERANCE_MS" in w for w in check_combinations(cfg))


def test_deaf_threshold_warns() -> None:
    cfg = Config(vad_abs_threshold=0.4)
    assert any("VAD_ABS_THRESHOLD" in w for w in check_combinations(cfg))


def test_reply_cap_above_what_the_token_budget_allows_warns() -> None:
    cfg = Config(reply_max_chars=5000, max_tokens=350)
    assert any("CHAT_MAX_TOKENS" in w for w in check_combinations(cfg))


# ──────────────────────────────────────────── ค่าเริ่มต้นต้องสะอาด (AC-19.4)
def env_example() -> dict[str, str]:
    out = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def test_env_example_produces_no_warnings(clean_env) -> None:
    """AC-19.4 — ถ้าค่าตัวอย่างยังเตือน ผู้ใช้จะเรียนรู้ว่า WARNING ไม่สำคัญ"""
    for key, value in env_example().items():
        clean_env.setenv(key, value)
    cfg = load_config()
    assert cfg.warnings == [], f"ค่าตัวอย่างใน .env.example ยังเตือน: {cfg.warnings}"


def test_bare_defaults_produce_no_warnings(clean_env) -> None:
    assert load_config().warnings == []


def test_every_numeric_key_in_env_example_has_a_range() -> None:
    """กันเพิ่ม key ใหม่แล้วลืมประกาศช่วง"""
    from web.server import WEB_RANGES      # key ฝั่งเว็บประกาศช่วงไว้ที่ web/server.py

    numeric = {k: v for k, v in env_example().items()
               if re.fullmatch(r"-?\d+(\.\d+)?", v)}
    skip = {"LOG_TRANSCRIPT", "ECHO_GUARD", "TTS_SINGLE_REQUEST",
            "TTS_SEARCH_FILLER", "WEB_SEARCH", "WEB_TRUST_PROXY"}   # ค่าเปิด/ปิด
    declared = set(RANGES) | set(WEB_RANGES)
    missing = sorted(k for k in numeric if k not in declared and k not in skip)
    assert missing == [], f"key ตัวเลขที่ยังไม่มีช่วง: {missing}"


def test_defaults_in_code_sit_inside_their_own_ranges() -> None:
    """ค่าปริยายในโค้ดต้องไม่ขัดกับช่วงที่ตัวเองประกาศ"""
    cfg = Config()
    pairs = {
        "CHAT_TEMPERATURE": cfg.temperature, "CHAT_MAX_TOKENS": cfg.max_tokens,
        "HTTP_RETRY_MAX": cfg.http_retry_max,
        "HTTP_RETRY_BASE_MS": cfg.http_retry_base_ms,
        "REPLY_MAX_SENTENCES": cfg.reply_max_sentences,
        "REPLY_MAX_CHARS": cfg.reply_max_chars, "KEEP_FINDINGS": cfg.keep_findings,
        "MIC_GAIN": cfg.mic_gain, "MIC_TARGET_NOISE": cfg.mic_target_noise,
        "VAD_ABS_THRESHOLD": cfg.vad_abs_threshold,
        "VAD_NOISE_MULT": cfg.vad_noise_mult, "VAD_CONFIRM_MS": cfg.vad_confirm_ms,
        "VAD_CONFIRM_MS_PLAYBACK": cfg.vad_confirm_ms_playback,
        "VAD_END_SILENCE_MS": cfg.vad_end_silence_ms,
        "VAD_MIN_UTTERANCE_MS": cfg.vad_min_utterance_ms,
        "VAD_MAX_UTTERANCE_MS": cfg.vad_max_utterance_ms,
        "ECHO_MARGIN": cfg.echo_margin,
        "BARGE_IN_MIN_CHARS": cfg.barge_in_min_chars,
        "TTS_MAX_CHARS": cfg.tts_max_chars, "TTS_FIRST_CHARS": cfg.tts_first_chars,
        "TTS_CHUNK_CHARS": cfg.tts_chunk_chars,
        "TTS_CHUNK_GROWTH": cfg.tts_chunk_growth,
        "SEARCH_RESULTS": cfg.search_results, "SEARCH_TIMEOUT": cfg.search_timeout,
        "FETCH_MAX_CHARS": cfg.fetch_max_chars,
        "FETCH_MAX_BYTES": cfg.fetch_max_bytes, "TOOL_ROUNDS": cfg.tool_rounds,
        "LOG_RETENTION_DAYS": cfg.log_retention_days,
    }
    assert set(pairs) == set(RANGES), "ตาราง RANGES กับค่าปริยายไม่ตรงกัน"
    for key, value in pairs.items():
        lo, hi, _ = RANGES[key]
        assert lo <= value <= hi, f"ค่าปริยายของ {key} ({value}) อยู่นอกช่วง {lo}-{hi}"


def test_warning_is_printed_to_stderr(clean_env, capsys) -> None:
    """AC-19.2 — ผู้ใช้ต้องเห็นตอนเปิดโปรแกรม ไม่ใช่เห็นแค่ในเทสต์"""
    clean_env.setenv("CHAT_TEMPERATURE", "9")
    load_config()
    assert "WARNING" in capsys.readouterr().err
