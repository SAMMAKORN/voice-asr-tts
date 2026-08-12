/* Voice Link — ฝั่งเบราว์เซอร์
   หน้าที่: จับเสียงไมค์ส่งขึ้นเซิร์ฟเวอร์, เล่นเสียงที่ TTS ส่งกลับแบบต่อเนื่อง
   (และตัดกลางคันได้ตอนถูกพูดแทรก), วาด caption/มิเตอร์, และคุมคู่มือเริ่มต้น */
'use strict';

const $ = (id) => document.getElementById(id);
const TOUR_KEY = 'voicelink.tour.v1';
const THEME_KEY = 'voicelink.theme.v1';

/* token ที่เซิร์ฟเวอร์ฝังมากับหน้าเว็บ (P1-1) — ต้องแนบไปทุกคำขอ
   ทั้ง WebSocket (?token=) และ /api/* (header X-Session-Token)
   เว็บอื่นอ่านค่านี้ไม่ได้เพราะติด Same-Origin Policy ของ fetch/XHR */
const TOKEN = (document.querySelector('meta[name="session-token"]') || {}).content || '';
const authHeaders = () => (TOKEN ? { 'X-Session-Token': TOKEN } : {});

let CFG = { mic_sr: 16000, speaker_sr: 24000, frame_ms: 20 };

// ───────────────────────────────────────────────────────── สถานะรวม
const A = {          // เครื่องเสียงบนเบราว์เซอร์
  ctx: null, stream: null, source: null, worklet: null,
  micAn: null, outAn: null, gain: null, ready: false,
};
const P = { next: 0, active: new Set() };   // คิวเล่นเสียง
let outLoopStarted = false;
let ws = null;
let live = false;             // ต่อ WebSocket และเริ่มสตรีมแล้ว
let msgEl = null;             // ฟองข้อความที่กำลังสตรีมอยู่
let turns = 0, interrupts = 0;

// ตัวนับสำหรับดูสถานะเสียงจริง (ใช้ตอนดีบักและตอนทดสอบอัตโนมัติ)
const VL = window.__vl = {
  chunks: 0, played: 0, cut: 0, stops: 0, sr: 0,
  playing: () => P.active.size,     // จำนวนก้อนเสียงที่จองคิวเล่นอยู่ตอนนี้
};
let thinkAt = 0;              // เวลาที่ AI เริ่มคิด — ใช้วัดว่ากว่าจะเห็นตัวอักษรแรกนานแค่ไหน

// ───────────────────────────────────────────────────────── UI พื้นฐาน
function setLink(state, text) {
  $('link').dataset.on = state;
  $('link-text').textContent = text;
}

function setStatus(now, hint) {
  $('status-now').textContent = now;
  if (hint !== undefined) $('status-hint').textContent = hint;
}

const ORB_BY_ICON = {
  '🎙': 'idle', '🎚': 'thinking', '📝': 'thinking',
  '💭': 'thinking', '🔊': 'speaking',
};

function setOrb(state) { $('orb').dataset.state = state; }

function addLog(text, level) {
  const box = $('logs');
  const el = document.createElement('div');
  el.className = level || '';
  const t = new Date().toTimeString().slice(0, 8);
  el.textContent = `${t}  ${text}`;
  box.appendChild(el);
  while (box.children.length > 200) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

function controls(on) {
  for (const id of ['btn-stop', 'btn-mute', 'btn-echo', 'btn-clear', 'input', 'btn-send']) {
    $(id).disabled = !on;
  }
}

// ───────────────────────────────────────────────────────── ธีมหน้าจอ
/* ธีมถูก apply ไปแล้วโดย inline script ใน <head> (กันหน้าจอกระพริบ)
   ที่นี่ดูแลแค่ปุ่มสลับ, การจำค่าที่เลือก และการตามธีมของระบบ */
const THEMES = {
  hud:  { icon: '🛰️', name: 'HUD' },
  clay: { icon: '🧸', name: 'ดินน้ำมัน' },
};

const isTheme = (name) => Object.prototype.hasOwnProperty.call(THEMES, name);

function currentTheme() {
  return document.documentElement.dataset.theme === 'clay' ? 'clay' : 'hud';
}

/* ยังไม่มีค่าใน localStorage = ผู้ใช้ยังไม่เคยเลือกเอง → ให้ตามระบบไปก่อน */
function themeChosenByUser() {
  try { return isTheme(localStorage.getItem(THEME_KEY)); } catch (_) { return false; }
}

function syncThemeButton() {
  const t = THEMES[currentTheme()];
  const icon = $('theme-icon'), name = $('theme-name');
  if (icon) icon.textContent = t.icon;
  if (name) name.textContent = t.name;
  const btn = $('btn-theme');
  if (btn) btn.title = `ธีมปัจจุบัน: ${t.name} — กดเพื่อสลับ`;
}

function applyTheme(name, persist) {
  const key = isTheme(name) ? name : 'hud';
  document.documentElement.dataset.theme = key;
  syncThemeButton();
  if (persist !== false) {
    try { localStorage.setItem(THEME_KEY, key); } catch (_) { /* localStorage ถูกปิด */ }
  }
  refreshWaveColors();               // สีเส้นคลื่นเปลี่ยนตามธีม
  addLog(`เปลี่ยนธีมเป็น ${THEMES[key].icon} ${THEMES[key].name}`);
}

// ───────────────────────────────────────────────────────── บทสนทนา
function beginMsg(role, label) {
  const empty = $('empty');
  if (empty) empty.remove();
  endMsg();
  const el = document.createElement('div');
  el.className = 'msg live';
  el.dataset.role = role;
  el.innerHTML = '<div class="who"></div><div class="body"></div>';
  el.querySelector('.who').textContent = role === 'user' ? 'คุณ' : 'AI';
  $('stream').appendChild(el);
  msgEl = el;
  scrollStream();
}

function writeMsg(text) {
  if (!msgEl) beginMsg('assistant');
  const body = msgEl.querySelector('.body');
  body.textContent += text;
  scrollStream();
}

function endMsg(note) {
  if (!msgEl) return;
  msgEl.classList.remove('live');
  if (note) {
    const b = document.createElement('span');
    b.className = 'badge';
    b.textContent = note;
    msgEl.querySelector('.body').appendChild(b);
  }
  msgEl = null;
  turns++;
  $('turn-count').textContent = `${turns} ข้อความ`;
  scrollStream();
}

function scrollStream() {
  const s = $('stream');
  s.scrollTop = s.scrollHeight;
}

/* แหล่งที่มาของข้อมูลที่ค้นจากเน็ต — ติดใต้คำตอบล่าสุดของ AI */
function showSources(items) {
  if (!items || !items.length) return;
  // ต้องเกาะกับคำตอบล่าสุดเท่านั้น ไม่ใช่ข้อความ AI อันไหนก็ได้
  // (ไม่งั้นเวลาคำตอบยังไม่ขึ้น ชิปจะไปโผล่ใต้ข้อความทักทายแทน)
  const all = $('stream').querySelectorAll('.msg');
  const last = all[all.length - 1];
  let target = msgEl || (last && last.dataset.role === 'assistant' ? last : null);
  if (!target) {
    target = document.createElement('div');
    target.className = 'msg';
    target.dataset.role = 'assistant';
    target.innerHTML = '<div class="who">AI</div><div class="body"></div>';
    $('stream').appendChild(target);
  }

  let box = target.querySelector('.sources');
  if (!box) {
    box = document.createElement('div');
    box.className = 'sources';
    box.innerHTML = '<span class="cap">🔎 แหล่งที่มา</span>';
    target.querySelector('.body').appendChild(box);
  }
  const have = new Set([...box.querySelectorAll('a')].map((a) => a.href));
  for (const s of items) {
    // host มาจากเซิร์ฟเวอร์ เพราะ URL.hostname คืนโดเมนไทยเป็น punycode
    let host = s.host;
    if (!host) {
      try { host = new URL(s.url).hostname.replace(/^www\./, ''); } catch (_) { host = s.url; }
    }
    const a = document.createElement('a');
    a.href = s.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.title = s.title || s.url;
    a.textContent = host;
    if (!have.has(a.href)) box.appendChild(a);
  }
  scrollStream();
}

// ───────────────────────────────────────────────────────── มิเตอร์ + คลื่นเสียง
const scale = (v) => Math.max(0, Math.min(1, Math.sqrt(v / 0.35)));

/* สีเส้นคลื่นมาจาก token ของธีม — cache ไว้เพราะ getComputedStyle บังคับให้
   เบราว์เซอร์คำนวณ style ใหม่ ถ้าเรียกทุกเฟรมก็คือ 60 ครั้ง/วินาที */
const WAVE = { line: '#5eeaff', axis: 'rgba(94,234,255,.12)' };

function refreshWaveColors() {
  const cs = getComputedStyle(document.documentElement);
  const line = cs.getPropertyValue('--wave-line').trim();
  const axis = cs.getPropertyValue('--wave-axis').trim();
  if (line) WAVE.line = line;
  if (axis) WAVE.axis = axis;
}

function drawWave() {
  const cv = $('wave');
  const g = cv.getContext('2d');
  const n = A.micAn ? A.micAn.frequencyBinCount : 0;
  const buf = n ? new Float32Array(n) : null;

  (function frame() {
    requestAnimationFrame(frame);
    const w = cv.width, h = cv.height;
    g.clearRect(0, 0, w, h);

    g.strokeStyle = WAVE.axis;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();

    if (!buf || !A.micAn) return;
    A.micAn.getFloatTimeDomainData(buf);
    g.strokeStyle = WAVE.line;
    g.globalAlpha = live ? 1 : 0.3;      // ยังไม่เริ่มสตรีม → เส้นจาง ๆ
    g.lineWidth = 1.6;
    g.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = h / 2 - buf[i] * (h / 2) * 2.4;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    g.globalAlpha = 1;
  })();
}

// ───────────────────────────────────────────────────────── เครื่องเสียง
async function ensureAudio() {
  if (A.ready) {
    if (A.ctx.state === 'suspended') await A.ctx.resume();
    return;
  }
  A.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,     // ให้เบราว์เซอร์ตัดเสียงลำโพงย้อนเข้าไมค์
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  A.ctx = new (window.AudioContext || window.webkitAudioContext)();
  await A.ctx.resume();
  await A.ctx.audioWorklet.addModule('/static/mic-worklet.js');

  A.source = A.ctx.createMediaStreamSource(A.stream);
  A.micAn = A.ctx.createAnalyser();
  A.micAn.fftSize = 1024;
  A.micAn.smoothingTimeConstant = 0.6;
  A.source.connect(A.micAn);

  A.worklet = new AudioWorkletNode(A.ctx, 'mic-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      targetSr: CFG.mic_sr,
      frameSamples: Math.round(CFG.mic_sr * CFG.frame_ms / 1000),
    },
  });
  A.worklet.port.onmessage = (e) => {
    if (live && ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  A.source.connect(A.worklet);
  // ต่อออกลำโพงผ่าน gain 0 — บางเบราว์เซอร์หยุดประมวลผล worklet ที่ไม่ได้ต่อปลายทาง
  const mute = A.ctx.createGain();
  mute.gain.value = 0;
  A.worklet.connect(mute).connect(A.ctx.destination);

  // เส้นทางเสียงขาออก (TTS)
  A.gain = A.ctx.createGain();
  A.outAn = A.ctx.createAnalyser();
  A.outAn.fftSize = 512;
  A.gain.connect(A.outAn);
  A.gain.connect(A.ctx.destination);

  A.ready = true;
  drawWave();
  micMeterLoop();
}

/* รายงานความดังของเสียงที่กำลังเล่น ให้เซิร์ฟเวอร์ใช้กันไมค์ได้ยินเสียงตัวเอง */
function outLevelLoop() {
  if (outLoopStarted || !A.outAn) return;
  outLoopStarted = true;
  const buf = new Float32Array(A.outAn.fftSize);
  setInterval(() => {
    if (!live || !ws || ws.readyState !== WebSocket.OPEN) return;
    let rms = 0;
    if (P.active.size || P.next > A.ctx.currentTime) {
      A.outAn.getFloatTimeDomainData(buf);
      let s = 0;
      for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
      rms = Math.sqrt(s / buf.length);
    }
    ws.send(JSON.stringify({ type: 'level', out: rms }));
  }, 100);
}

/* มิเตอร์ในหน้าคู่มือ (ทำงานได้ก่อนเชื่อมต่อเซิร์ฟเวอร์) */
function micMeterLoop() {
  const buf = new Float32Array(A.micAn.fftSize);
  setInterval(() => {
    A.micAn.getFloatTimeDomainData(buf);
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    const rms = Math.sqrt(s / buf.length);
    const fill = $('mic-fill');
    if (fill) fill.style.width = (scale(rms) * 100).toFixed(1) + '%';
  }, 60);
}

// เล่นเสียงจาก TTS แบบต่อคิวไม่ให้ขาดช่วง
function playChunk(epoch, seq, i16) {
  if (!A.ctx) return;
  const buf = A.ctx.createBuffer(1, i16.length, CFG.speaker_sr);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;

  const src = A.ctx.createBufferSource();
  src.buffer = buf;
  src.connect(A.gain);

  const at = Math.max(A.ctx.currentTime + 0.04, P.next);
  src.start(at);
  P.next = at + buf.duration;
  P.active.add(src);
  VL.chunks++;
  VL.sr = A.ctx.sampleRate;

  src.onended = () => {
    P.active.delete(src);
    VL.played++;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'played', epoch, seq }));
    }
  };
}

// ตัดเสียงทั้งหมดทันที (ถูกพูดแทรก / กดหยุด)
function stopPlayback() {
  if (P.active.size) VL.cut++;
  for (const src of P.active) {
    src.onended = null;               // ไม่ต้องรายงานว่าเล่นจบ เพราะโดนตัด
    try { src.stop(); } catch (_) { /* หยุดไปแล้ว */ }
  }
  P.active.clear();
  P.next = 0;
}

// ───────────────────────────────────────────────────────── WebSocket
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(TOKEN)}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    live = true;
    setLink('1', 'เชื่อมต่อแล้ว');
    setStatus('กำลังเตรียมระบบ...', 'วัดเสียงรบกวนรอบข้าง อยู่เงียบ ๆ ครู่หนึ่ง');
    outLevelLoop();
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') return handleJson(JSON.parse(ev.data));
    const dv = new DataView(ev.data);
    const epoch = dv.getUint32(0, true);
    const seq = dv.getUint32(4, true);
    playChunk(epoch, seq, new Int16Array(ev.data, 8));
  };

  ws.onclose = () => {
    live = false;
    stopPlayback();
    setLink('0', 'หลุดการเชื่อมต่อ');
    setStatus('การเชื่อมต่อหลุด', 'กด “เริ่มระบบ” อีกครั้งเพื่อเชื่อมต่อใหม่');
    setOrb('idle');
    controls(false);
    $('btn-start').disabled = false;
    $('btn-start').textContent = 'เชื่อมต่อใหม่';
  };

  ws.onerror = () => addLog('เชื่อมต่อ WebSocket ไม่สำเร็จ', 'error');
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function handleJson(m) {
  switch (m.type) {
    case 'ready':
      CFG.speaker_sr = m.speaker_sr || CFG.speaker_sr;
      $('chip-llm').textContent = m.chat_model;
      $('chip-asr').textContent = m.asr_model;
      $('chip-tts').textContent = m.tts_model;
      $('btn-mute').dataset.on = m.tts_enabled ? '0' : '1';
      $('btn-echo').dataset.on = m.echo_guard ? '1' : '0';
      controls(true);
      $('btn-start').textContent = '● ทำงานอยู่';
      $('btn-start').disabled = true;
      // ไม่แสดง path ของ log แล้ว — เซิร์ฟเวอร์ไม่ส่งออกมาให้ client อีกต่อไป (P1-1)
      addLog('พร้อมใช้งาน · บันทึกบทสนทนาไว้ที่เครื่องที่รันเซิร์ฟเวอร์', 'good');
      break;

    case 'calibrated': {
      addLog(`เสียงรบกวน ${m.noise} · เกณฑ์เริ่มอัด ${m.threshold}`);
      $('meter-mark').style.insetInlineStart = (scale(m.threshold) * 100).toFixed(1) + '%';
      break;
    }

    case 'status':
      if (m.text) {
        setStatus(m.text, '');
        setOrb(ORB_BY_ICON[m.icon] || 'idle');
        if (m.icon === '💭') thinkAt = performance.now();
      }
      break;

    case 'speech':
      if (m.state === 'start') { setOrb('listening'); setStatus('กำลังฟังคุณพูด...', ''); }
      break;

    case 'level': {
      const el = $('meter-fill');
      el.style.width = (scale(m.rms) * 100).toFixed(1) + '%';
      $('meter').classList.toggle('hot', m.rms > m.threshold);
      $('meter-mark').style.insetInlineStart = (scale(m.threshold) * 100).toFixed(1) + '%';
      $('meter-val').textContent = m.rms.toFixed(3);
      break;
    }

    case 'begin':  beginMsg(m.role, m.label); break;

    case 'delta':
      // ตัวอักษรแรกของ AI = จุดที่ผู้ใช้เห็นว่ามันเริ่มตอบ (รวมเวลาส่งกลับมาแล้ว)
      if (thinkAt && msgEl && msgEl.dataset.role === 'assistant') {
        $('stat-llm').textContent = Math.round(performance.now() - thinkAt) + ' ms';
        thinkAt = 0;
        setStatus('AI กำลังตอบ', 'พูดแทรกได้เลย ไม่ต้องรอให้พูดจบ');
        setOrb('speaking');
      }
      writeMsg(m.text);
      break;

    case 'end':    endMsg(m.interrupted ? 'ถูกพูดขัด' : ''); break;

    case 'sources': showSources(m.items); break;

    case 'stop':   VL.stops++; stopPlayback(); break;

    case 'metric': applyMetric(m); break;

    case 'setting':
      if (m.key === 'mute') $('btn-mute').dataset.on = m.on ? '1' : '0';
      if (m.key === 'echo_guard') $('btn-echo').dataset.on = m.on ? '1' : '0';
      break;

    case 'cleared':
      $('stream').innerHTML = '';
      turns = 0;
      $('turn-count').textContent = '0 ข้อความ';
      addLog('ล้างประวัติการสนทนาแล้ว');
      break;

    case 'log':
      addLog(m.text, m.level === 'info' ? '' : m.level);
      break;

    case 'closed':
      setLink('0', 'ปิด session แล้ว');
      break;
  }
}

function applyMetric(m) {
  if (m.kind === 'asr' && m.latency_ms != null) {
    $('stat-asr').textContent = m.latency_ms + ' ms';
  } else if (m.kind === 'tts' && m.latency_ms != null) {
    $('stat-tts').textContent = m.latency_ms + ' ms';
  } else if (m.kind === 'tool') {
    addLog(`ค้นเน็ต: ${m.arg || ''} (${m.latency_ms} ms)`, 'good');
  } else if (m.kind === 'interrupt') {
    interrupts++;
    $('stat-int').textContent = String(interrupts);
  } else if (m.kind && m.kind.endsWith('_error')) {
    addLog(`${m.kind}: ${m.error || ''}`, 'error');
  }
}

// ───────────────────────────────────────────────────────── เริ่ม/หยุดระบบ
async function start() {
  const btn = $('btn-start');
  btn.disabled = true;
  btn.textContent = 'กำลังเริ่ม...';
  try {
    await ensureAudio();
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'เริ่มระบบ';
    setStatus('เปิดไมโครโฟนไม่ได้', 'ตรวจสิทธิ์ไมค์ของเบราว์เซอร์แล้วลองใหม่');
    addLog('ขอสิทธิ์ไมโครโฟนไม่สำเร็จ: ' + err.message, 'error');
    return;
  }
  btn.textContent = 'กำลังเชื่อมต่อ...';
  connect();
}

// ───────────────────────────────────────────────────────── ปุ่มต่าง ๆ
$('btn-start').addEventListener('click', start);

$('btn-stop').addEventListener('click', () => {
  stopPlayback();
  send({ type: 'interrupt' });
});

$('btn-mute').addEventListener('click', (e) => {
  const on = e.currentTarget.dataset.on !== '1';   // on = ปิดเสียง
  e.currentTarget.dataset.on = on ? '1' : '0';
  e.currentTarget.textContent = on ? '🔇 ปิดเสียง' : '🔊 เสียง';
  if (on) stopPlayback();
  send({ type: 'mute', on });
});

$('btn-echo').addEventListener('click', (e) => {
  const on = e.currentTarget.dataset.on !== '1';
  e.currentTarget.dataset.on = on ? '1' : '0';
  send({ type: 'echo_guard', on });
  addLog(on ? 'เปิดระบบกันเสียงลำโพงย้อนเข้าไมค์ (โหมดลำโพง)'
            : 'ปิดระบบกันเสียงลำโพง — พูดแทรกไวสุด (โหมดหูฟัง)');
});

$('btn-clear').addEventListener('click', () => send({ type: 'clear' }));
$('btn-help').addEventListener('click', () => openTour(0));

// กดเองแล้วถือว่า “เลือกเอง” — จำค่าไว้และเลิกตามธีมของระบบ
$('btn-theme').addEventListener('click', () => {
  applyTheme(currentTheme() === 'hud' ? 'clay' : 'hud');
});

$('compose').addEventListener('submit', (e) => {
  e.preventDefault();
  const el = $('input');
  const text = el.value.trim();
  if (!text) return;
  el.value = '';
  send({ type: 'text', text });
});

// ───────────────────────────────────────────────────────── คู่มือเริ่มต้น
const TOUR_LAST = 4;
let step = 0;

function renderSteps() {
  const box = $('tour-steps');
  box.innerHTML = '';
  for (let i = 0; i <= TOUR_LAST; i++) {
    const i2 = document.createElement('i');
    if (i === step) i2.className = 'on';
    else if (i < step) i2.className = 'done';
    box.appendChild(i2);
  }
}

function showStep(n) {
  step = Math.max(0, Math.min(TOUR_LAST, n));
  for (const s of document.querySelectorAll('#tour-body section')) {
    s.hidden = Number(s.dataset.step) !== step;
  }
  $('tour-prev').disabled = step === 0;
  $('tour-next').textContent = step === TOUR_LAST ? 'เริ่มใช้งาน' : 'ถัดไป';
  $('tour-body').scrollTop = 0;
  renderSteps();
}

function openTour(n) {
  $('overlay').hidden = false;
  showStep(n || 0);
}

function closeTour() {
  $('overlay').hidden = true;
  localStorage.setItem(TOUR_KEY, '1');
}

$('tour-prev').addEventListener('click', () => showStep(step - 1));

$('tour-next').addEventListener('click', () => {
  if (step === TOUR_LAST) {
    closeTour();
    if (!live) start();
  } else {
    showStep(step + 1);
  }
});

$('tour-skip').addEventListener('click', closeTour);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('overlay').hidden) closeTour();
});

// ขั้นที่ 1 — ขอสิทธิ์ไมค์
$('btn-mic').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = 'กำลังขอสิทธิ์...';
  try {
    await ensureAudio();
    $('mic-state').textContent = 'เปิดแล้ว · ลองพูดดู';
    btn.textContent = '✓ ไมโครโฟนพร้อม';
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'ลองอีกครั้ง';
    $('mic-state').textContent = 'ไม่ได้รับสิทธิ์';
    addLog('ขอสิทธิ์ไมโครโฟนไม่สำเร็จ: ' + err.message, 'error');
  }
});

// ขั้นที่ 2 — ตรวจระบบ
$('btn-check').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const box = $('check');
  btn.disabled = true;
  btn.textContent = 'กำลังตรวจ...';
  box.innerHTML = '';
  const wait = row('…', 'ระบบ', 'กำลังเรียก TTS → ASR → LLM ตามลำดับ', '');
  box.appendChild(wait);
  try {
    const res = await fetch('/api/selftest', { method: 'POST', headers: authHeaders() });
    const data = await res.json();
    box.innerHTML = '';
    for (const s of data.steps) {
      box.appendChild(row(s.ok ? '✓' : '✗', s.name, s.detail,
                          s.ms ? s.ms + ' ms' : '', s.ok, s.model));
    }
    btn.textContent = data.ok ? '✓ ระบบพร้อม' : 'ตรวจอีกครั้ง';
    btn.disabled = data.ok;
  } catch (err) {
    box.innerHTML = '';
    box.appendChild(row('✗', 'ระบบ', 'เรียกเซิร์ฟเวอร์ไม่ได้: ' + err.message, '', false));
    btn.disabled = false;
    btn.textContent = 'ตรวจอีกครั้ง';
  }
});

function row(icon, name, detail, ms, ok, model) {
  const el = document.createElement('div');
  el.className = 'check-row';
  if (ok !== undefined) el.dataset.ok = ok ? '1' : '0';
  el.innerHTML = '<div class="icon"></div><div class="name"></div>' +
                 '<div class="detail"></div><div class="ms"></div>';
  el.querySelector('.icon').textContent = icon;
  el.querySelector('.name').textContent = name;
  el.querySelector('.detail').textContent = model ? `${model} — ${detail}` : detail;
  el.querySelector('.ms').textContent = ms;
  return el;
}

// ───────────────────────────────────────────────────────── เริ่มต้นหน้า
(async function init() {
  // ธีมถูกเซ็ตจาก inline script ใน <head> แล้ว — ตรงนี้แค่ทำให้ป้ายบนปุ่มตรงกัน
  // (ห้ามเรียก applyTheme ตอนเริ่ม ไม่งั้นค่าที่ผู้ใช้เลือกไว้จะถูกเขียนทับ)
  syncThemeButton();
  refreshWaveColors();               // drawWave ถูกเรียกทีหลัง จึงต้องมีสีไว้ก่อน

  // ยังไม่เคยเลือกเอง → เปลี่ยนตามธีมของระบบ (ไม่บันทึกลง localStorage)
  const mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)');
  if (mq && mq.addEventListener) {
    mq.addEventListener('change', (e) => {
      if (themeChosenByUser()) return;
      applyTheme(e.matches ? 'clay' : 'hud', false);
    });
  }

  try {
    const res = await fetch('/api/config', { headers: authHeaders() });
    const data = await res.json();
    CFG = Object.assign(CFG, data);
    $('chip-llm').textContent = data.chat_model;
    $('chip-asr').textContent = data.asr_model;
    $('chip-tts').textContent = data.tts_model;
  } catch (_) {
    addLog('อ่านค่าตั้งจากเซิร์ฟเวอร์ไม่ได้', 'warn');
  }

  if (localStorage.getItem(TOUR_KEY)) {
    $('overlay').hidden = true;
  } else {
    openTour(0);
  }
})();
