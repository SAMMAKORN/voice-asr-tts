"""เสียงอ้างอิงสำหรับโคลนเสียง (vc/voiceclone.py)

เน้นกฎที่พังเงียบ ๆ ถ้าทำผิด: ต้องเป็น data URI, ต้องมี ref_text คู่เสมอ,
ไฟล์อ้างอิงที่เสียต้องไม่ทำให้พูดไม่ได้ทั้งระบบ และความดังต้องถูกปรับก่อนส่ง
"""
from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from vc.voiceclone import (MIN_REF_SEC, ReferenceError, VoiceReference, prepare,
                           read_wav_mono, resolve_path, sidecar, to_data_uri,
                           write_wav_mono)

pytestmark = pytest.mark.unit
SR = 16000


def tone(seconds: float, amp: float = 0.5, sr: int = SR, freq: float = 150.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    return write_wav_mono(np.sin(2 * np.pi * freq * t) * amp, sr)


def wav_of(data: bytes) -> tuple[np.ndarray, int]:
    return read_wav_mono(data)


# --- รูปแบบที่เซิร์ฟเวอร์ยอมรับ ------------------------------------------------
def test_data_uri_prefix_is_required_shape():
    """base64 เปล่า ๆ ทำให้เซิร์ฟเวอร์ตอบ 500 — ต้องมี data: นำหน้าเสมอ"""
    uri = to_data_uri(tone(2.0))
    assert uri.startswith("data:audio/wav;base64,")
    body = uri.split(",", 1)[1]
    assert base64.b64decode(body)[:4] == b"RIFF"


def test_prepare_keeps_wav_readable_and_mono():
    out = prepare(tone(2.0))
    x, sr = wav_of(out)
    assert sr == SR and x.ndim == 1


def test_stereo_reference_is_downmixed():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros(SR * 2 * 2, dtype="<i2").tobytes())
    x, _ = read_wav_mono(buf.getvalue())
    assert x.ndim == 1 and x.size == SR * 2


# --- ความดัง -----------------------------------------------------------------
def test_quiet_reference_is_normalised():
    """ไฟล์อ้างอิงเบา = AI พูดเบาตาม (วัดจริง rms 0.029 → เสียงที่ได้ 0.017)"""
    quiet = tone(2.0, amp=0.03)
    before, _ = wav_of(quiet)
    after, _ = wav_of(prepare(quiet))
    assert np.abs(before).max() < 0.05
    assert np.abs(after).max() > 0.9


def test_normalise_can_be_turned_off():
    quiet = tone(2.0, amp=0.03)
    after, _ = wav_of(prepare(quiet, normalize=False))
    assert np.abs(after).max() < 0.05


def test_silent_reference_does_not_divide_by_zero():
    silent = write_wav_mono(np.zeros(SR * 2, dtype=np.float32), SR)
    out, _ = wav_of(prepare(silent))
    assert np.abs(out).max() == 0.0


# --- ความยาว -----------------------------------------------------------------
def test_reference_shorter_than_minimum_is_rejected():
    with pytest.raises(ReferenceError):
        prepare(tone(MIN_REF_SEC / 2))


def test_max_sec_trims():
    x, sr = wav_of(prepare(tone(8.0), max_sec=3.0))
    assert 2.9 < x.size / sr < 3.1


def test_max_sec_zero_keeps_whole_file():
    x, sr = wav_of(prepare(tone(8.0), max_sec=0.0))
    assert x.size / sr > 7.9


def test_empty_and_broken_wav_are_reference_errors():
    with pytest.raises(ReferenceError):
        prepare(write_wav_mono(np.zeros(0, dtype=np.float32), SR))
    with pytest.raises(ReferenceError):
        read_wav_mono(b"not a wav at all")


# --- VoiceReference ----------------------------------------------------------
def test_payload_carries_both_keys(tmp_path):
    """ส่ง ref_audio โดยไม่มี ref_text = เซิร์ฟเวอร์ตอบ 200 แต่ได้เสียงแทบเงียบ"""
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    ref = VoiceReference(str(wav), text="สวัสดี")
    p = ref.payload()
    assert set(p) == {"ref_audio", "ref_text"}
    assert p["ref_text"] == "สวัสดี"
    assert p["ref_audio"].startswith("data:audio/wav;base64,")


def test_text_falls_back_to_sidecar_file(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    sidecar(wav).write_text("คำอ่านจากไฟล์\n", encoding="utf-8")
    assert VoiceReference(str(wav)).payload()["ref_text"] == "คำอ่านจากไฟล์"


def test_text_falls_back_to_asr_and_is_cached(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    calls = []

    def fake_asr(pcm, sr):
        calls.append((pcm.dtype, sr))
        return "ถอดมาจาก ASR"

    ref = VoiceReference(str(wav))
    assert ref.payload(fake_asr)["ref_text"] == "ถอดมาจาก ASR"
    assert len(calls) == 1 and calls[0][1] == SR
    # เรียกซ้ำต้องไม่ยิง ASR อีก และต้องเขียน .txt ไว้ให้รอบหน้า
    assert ref.payload(fake_asr)["ref_text"] == "ถอดมาจาก ASR"
    assert len(calls) == 1
    assert sidecar(wav).read_text(encoding="utf-8").strip() == "ถอดมาจาก ASR"


def test_missing_text_without_asr_disables_cloning(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    ref = VoiceReference(str(wav))
    assert ref.payload() is None
    assert ref.error


def test_missing_file_disables_cloning_instead_of_raising(tmp_path):
    ref = VoiceReference(str(tmp_path / "nope.wav"), text="x")
    assert ref.payload() is None
    assert "nope.wav" in ref.error or ref.error


def test_broken_file_is_remembered_and_not_retried(tmp_path):
    wav = tmp_path / "bad.wav"
    wav.write_bytes(b"garbage")
    ref = VoiceReference(str(wav), text="x")
    assert ref.payload() is None
    first = ref.error
    assert ref.payload() is None and ref.error == first


def test_relative_paths_resolve_against_repo_root():
    assert resolve_path("sound/thai_default.wav").is_absolute()
    assert resolve_path("/tmp/x.wav") == resolve_path("/tmp/x.wav")


def test_shipped_reference_loads_with_its_sidecar():
    """ไฟล์ที่แถมมาต้องใช้ได้ทันทีโดยไม่ต้องต่อเน็ตไปถอดเสียง"""
    ref = VoiceReference("sound/thai_default.wav")
    p = ref.payload()          # ไม่ส่ง ASR เข้าไป — ต้องได้จาก .txt ข้าง ๆ
    assert p is not None, ref.error
    assert p["ref_text"].strip()


# --- ต่อเข้ากับ ApiClient -----------------------------------------------------
def _mock_api(cfg, seen: list[dict]):
    """ApiClient ที่ดัก payload ของ /v1/audio/speech ไว้ดูโดยไม่ยิงเน็ตจริง"""
    import httpx

    from vc.api import ApiClient, pcm16_to_wav

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        if request.url.path.endswith("/audio/speech"):
            seen.append(_json.loads(request.content))
            return httpx.Response(200, content=pcm16_to_wav(np.zeros(160, np.int16), SR),
                                  headers={"content-type": "audio/wav"})
        return httpx.Response(200, json={"text": "ถอดมาจาก ASR"})

    api = ApiClient(cfg)
    transport = httpx.MockTransport(handler)
    for name in ("_audio", "_chat"):
        old = getattr(api, name)
        setattr(api, name, httpx.Client(base_url=old.base_url, transport=transport,
                                        timeout=old.timeout))
        old.close()
    return api


def test_synthesize_attaches_reference(cfg, tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    sidecar(wav).write_text("คำอ่าน", encoding="utf-8")
    cfg.tts_ref_audio = str(wav)
    seen: list[dict] = []
    _mock_api(cfg, seen).synthesize("สวัสดี", SR)
    assert seen[0]["ref_text"] == "คำอ่าน"
    assert seen[0]["ref_audio"].startswith("data:audio/wav;base64,")


def test_reference_is_encoded_once_and_reused(cfg, tmp_path):
    """payload หนักราว 290KB ต่อก้อน — เข้ารหัสใหม่ทุกก้อนคือของแพงที่ไม่จำเป็น"""
    wav = tmp_path / "ref.wav"
    wav.write_bytes(tone(2.0))
    sidecar(wav).write_text("คำอ่าน", encoding="utf-8")
    cfg.tts_ref_audio = str(wav)
    seen: list[dict] = []
    api = _mock_api(cfg, seen)
    for _ in range(3):
        api.synthesize("สวัสดี", SR)
    assert len({p["ref_audio"] for p in seen}) == 1
    assert api._voice_ref._uri is not None


def test_cloning_off_sends_plain_payload(cfg, tmp_path):
    cfg.tts_ref_audio = ""
    seen: list[dict] = []
    _mock_api(cfg, seen).synthesize("สวัสดี", SR)
    assert set(seen[0]) == {"model", "input"}


def test_broken_reference_still_speaks(cfg, tmp_path):
    """เสียงอ้างอิงพังต้องตกกลับไปพูดแบบไม่โคลน ไม่ใช่พูดไม่ได้เลย"""
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"garbage")
    cfg.tts_ref_audio = str(bad)
    seen: list[dict] = []
    out = _mock_api(cfg, seen).synthesize("สวัสดี", SR)
    assert out.size > 0
    assert set(seen[0]) == {"model", "input"}


def test_edge_backend_never_gets_reference(cfg):
    cfg.tts_backend = "edge"
    assert cfg.clone_enabled is False


def test_voice_gender_follows_reference(cfg):
    cfg.tts_ref_gender = "female"
    assert cfg.voice_gender == "female"
    cfg.tts_ref_audio = ""          # ไม่โคลน = เสียงสุ่ม ไม่มีเพศแน่นอน
    assert cfg.voice_gender == "male"
