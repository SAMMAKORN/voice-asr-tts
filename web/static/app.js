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
let sessions = 0;             // จำนวน session ที่เซิร์ฟเวอร์สร้างให้ (นับจาก ready)
let audioBlocked = '';        // เหตุผลที่ใช้ไมค์ไม่ได้เลย (เช่น ไม่ใช่ secure context)

// ตัวนับสำหรับดูสถานะเสียงจริง (ใช้ตอนดีบักและตอนทดสอบอัตโนมัติ)
const VL = window.__vl = {
  chunks: 0, played: 0, cut: 0, stops: 0, sr: 0,
  attempts: 0, reconnects: 0, orbWrites: 0,
  playing: () => P.active.size,     // จำนวนก้อนเสียงที่จองคิวเล่นอยู่ตอนนี้
};
let thinkAt = 0;              // เวลาที่ AI เริ่มคิด — ใช้วัดว่ากว่าจะเห็นตัวอักษรแรกนานแค่ไหน

const reducedMotion = () => window.matchMedia
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ───────────────────────────────────────────────────────── UI พื้นฐาน
function setLink(state, text) {
  $('link').dataset.on = state;
  $('link-text').textContent = text;
}

function setStatus(now, hint) {
  $('status-now').textContent = now;
  if (hint !== undefined) $('status-hint').textContent = hint;
}

/* ไอคอนสถานะจากเซิร์ฟเวอร์ → สถานะของวงแหวน
   ค่าที่ไม่รู้จักต้องกลายเป็น 'thinking' ไม่ใช่ 'idle' เพราะ 'idle' แปลว่า
   "พร้อมฟัง" ซึ่งหลอกผู้ใช้ว่าพูดได้ทั้งที่ระบบกำลังยุ่งอยู่ */
const ORB_BY_ICON = {
  '🎙': 'idle', '🎚': 'thinking', '📝': 'thinking',
  '💭': 'thinking', '🔎': 'searching', '🔊': 'speaking', '✂️': 'thinking',
};

/* phase จริงจากเซิร์ฟเวอร์ (P2-5) — แม่นกว่าเดาจากไอคอน */
const ORB_BY_PHASE = {
  idle: 'idle', transcribing: 'thinking',
  generating: 'thinking', speaking: 'speaking',
};

function setOrb(state) {
  const orb = $('orb');
  if (orb.dataset.state === state) return;
  orb.dataset.state = state;
  // ระหว่างคิด/ค้นเน็ต/ผิดพลาด CSS คุมวงแหวนเอง — เลิกขับด้วยระดับเสียง
  ORB.driven = state === 'idle' || state === 'listening' || state === 'speaking';
  if (!ORB.driven) ORB.target = 0;
}

// ───────────────────────────────────────── วงแหวนระดับเสียงรอบวงกลมไมค์ (P2-7)
/* markup + CSS ของ #orb-val มีมาตั้งแต่ต้นแต่ไม่มีโค้ดขับเลย (dead markup)
   ตรงนี้ต่อ event `level` เข้ากับ stroke-dashoffset ผ่าน requestAnimationFrame
   แล้วหน่วงค่าให้ลื่น (ค่าดิบเข้ามา ~10 ครั้ง/วินาที ถ้าเขียนตรง ๆ จะกระตุก) */
const ORB = { target: 0, shown: 0, len: 226.2, driven: true, raf: 0 };

function paintOrbRing(value) {
  const el = $('orb-val');
  if (!el) return;
  el.setAttribute('stroke-dashoffset', (ORB.len * (1 - value)).toFixed(1));
  VL.orbWrites++;
}

function orbLevel(rms) {
  ORB.target = Math.max(0, Math.min(1, scale(rms)));
  if (!ORB.driven) return;
  if (reducedMotion()) {          // ไม่ต้องหน่วงต่อเนื่อง เขียนครั้งเดียวพอ
    ORB.shown = ORB.target;
    paintOrbRing(ORB.shown);
    return;
  }
  startOrbLoop();
}

function startOrbLoop() {
  if (ORB.raf) return;
  const frame = () => {
    ORB.shown += (ORB.target - ORB.shown) * 0.25;
    if (ORB.driven) paintOrbRing(ORB.shown);
    if (Math.abs(ORB.target - ORB.shown) < 0.002 && ORB.target === 0) {
      ORB.raf = 0;                // นิ่งแล้ว หยุดวนเพื่อไม่กินแบตเปล่า ๆ
      return;
    }
    ORB.raf = requestAnimationFrame(frame);
  };
  ORB.raf = requestAnimationFrame(frame);
}

// ───────────────────────────────────── แจ้งเตือนที่เห็นได้ทุกขนาดจอ (P2-8/P2-14)
/* แผง error เดิมอยู่ใน .side ซึ่ง CSS ซ่อนทิ้งที่ความกว้าง ≤ 1000px
   ผู้ใช้มือถือจึงไม่เคยเห็น error จาก backend เลย */
function showAlert(text, opts) {
  const box = $('alert');
  $('alert-msg').textContent = text;
  box.hidden = false;
  box.dataset.sticky = (opts && opts.sticky) ? '1' : '0';
  if (!opts || opts.orb !== false) setOrb('error');
}

function clearAlert(force) {
  const box = $('alert');
  if (box.hidden) return;
  if (!force && box.dataset.sticky === '1') return;   // เช่น secure context
  box.hidden = true;
  $('alert-msg').textContent = '';
}

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

/* ประกาศให้ screen reader ครั้งเดียวตอนข้อความจบ (P3-16 / AC-16.1)
   #stream ตั้ง aria-live="off" ไว้ เพราะข้อความต่อทีละ token ถ้าประกาศที่นั่น
   ผู้ใช้ VoiceOver จะได้ยินย่อหน้าเดิมซ้ำทุกตัวอักษรจนฟังไม่รู้เรื่อง */
function announce(text) {
  const box = $('sr-live');
  if (!box || !text) return;
  // เขียนค่าเดิมทับค่าเดิม screen reader จะไม่ประกาศ — ล้างก่อนหนึ่งจังหวะ
  box.textContent = '';
  setTimeout(() => { box.textContent = text; }, 60);
}

function endMsg(note) {
  if (!msgEl) return;
  msgEl.classList.remove('live');
  const said = msgEl.querySelector('.body').textContent.trim();
  const who = msgEl.dataset.role === 'user' ? 'คุณพูดว่า' : 'AI ตอบว่า';
  if (said) announce(`${who} ${said}${note ? ` (${note})` : ''}`);
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

// ───────────────────────────────────────── เลื่อนตามแบบเกาะก้น (sticky bottom)
/* เดิม scrollStream() กระชากลงก้นทุก token ผู้ใช้จึงอ่านย้อนระหว่าง AI พิมพ์ไม่ได้เลย
   ตอนนี้เลื่อนตามเฉพาะเมื่อผู้ใช้ยังอยู่ใกล้ก้น (≤ 48px) ไม่งั้นขึ้นปุ่ม "↓ ล่าสุด" ให้กด */
const STICK_PX = 48;
let stickBottom = true;

function nearBottom() {
  const s = $('stream');
  return s.scrollHeight - s.scrollTop - s.clientHeight <= STICK_PX;
}

function showJump(on) {
  const btn = $('jump');
  if (btn) btn.hidden = !on;
}

function scrollStream() {
  const s = $('stream');
  if (stickBottom) {
    s.scrollTop = s.scrollHeight;
    showJump(false);
    return;
  }
  showJump(true);           // มีข้อความใหม่แต่ผู้ใช้กำลังอ่านย้อนอยู่
}

function jumpToLatest() {
  const s = $('stream');
  stickBottom = true;
  s.scrollTop = s.scrollHeight;
  showJump(false);
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
/* เชื่อมต่อใหม่เองแบบ exponential backoff — เดิมหลุดแล้วจบเลย ผู้ใช้ต้องกดปุ่มเอง
   ทุกครั้งที่เชื่อมต่อใหม่ เซิร์ฟเวอร์สร้าง WebSession ใหม่ = AI ลืมทุกอย่าง
   จึงต้องแทรกเส้นคั่นบอกให้ชัด ไม่ปล่อยให้ผู้ใช้เข้าใจผิดว่ามันยังจำได้ */
const RETRY_DELAYS = [500, 1000, 2000, 4000, 8000, 15000];
const R = { tries: 0, timer: 0, wanted: false };

function reconnectSoon() {
  if (!R.wanted || R.timer) return;
  if (R.tries >= RETRY_DELAYS.length) {
    setLink('0', 'เชื่อมต่อไม่ได้');
    setStatus('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้',
              'ลองใหม่อัตโนมัติครบแล้ว — กด “เชื่อมต่อใหม่” เพื่อลองอีกครั้ง');
    showAlert('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้หลังลองใหม่ '
              + RETRY_DELAYS.length + ' ครั้ง กดปุ่ม “เชื่อมต่อใหม่” เพื่อลองอีกครั้ง');
    $('btn-start').disabled = false;
    $('btn-start').textContent = 'เชื่อมต่อใหม่';
    return;
  }
  const wait = RETRY_DELAYS[R.tries];
  R.tries++;
  setLink('0', 'กำลังเชื่อมต่อใหม่...');
  setStatus('กำลังเชื่อมต่อใหม่...',
            `ครั้งที่ ${R.tries} จาก ${RETRY_DELAYS.length} — อีก ${Math.round(wait / 1000)} วินาที`);
  setOrb('thinking');
  addLog(`การเชื่อมต่อหลุด — จะลองใหม่ในอีก ${wait} ms (ครั้งที่ ${R.tries})`, 'warn');
  R.timer = setTimeout(() => { R.timer = 0; connect(); }, wait);
}

function connect() {
  R.wanted = true;
  VL.attempts++;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  // ส่งเสียงที่เลือก/จำไว้ไปตั้งแต่ตอนต่อ WebSocket กัน session ทักทายด้วยเสียง
  // ของ .env ไปก่อนแล้วค่อยสลับทีหลัง (พูดผิดเพศไปแล้วประโยคแรก)
  const voice = encodeURIComponent($('sel-voice').value || '');
  ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(TOKEN)}&voice=${voice}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    live = true;
    if (R.tries) VL.reconnects++;
    R.tries = 0;
    clearAlert();
    setLink('1', 'เชื่อมต่อแล้ว');
    setStatus('กำลังเตรียมระบบ...', 'วัดเสียงรบกวนรอบข้าง อยู่เงียบ ๆ ครู่หนึ่ง');
    outLevelLoop();
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      let m = null;
      try { m = JSON.parse(ev.data); } catch (_) { return; }
      return handleJson(m);
    }
    const dv = new DataView(ev.data);
    const epoch = dv.getUint32(0, true);
    const seq = dv.getUint32(4, true);
    playChunk(epoch, seq, new Int16Array(ev.data, 8));
  };

  ws.onclose = () => {
    live = false;
    stopPlayback();
    controls(false);
    setOrb('idle');
    if (R.wanted) {
      reconnectSoon();
      return;
    }
    setLink('0', 'หลุดการเชื่อมต่อ');
    setStatus('การเชื่อมต่อหลุด', 'กด “เชื่อมต่อใหม่” เพื่อเริ่มต่อใหม่');
    $('btn-start').disabled = false;
    $('btn-start').textContent = 'เชื่อมต่อใหม่';
  };

  ws.onerror = () => addLog('เชื่อมต่อ WebSocket ไม่สำเร็จ', 'error');
}

/* เส้นคั่นบอกว่าความจำเริ่มใหม่ — เรียกเมื่อได้ ready ของ session ที่สองขึ้นไป */
function markMemoryReset() {
  const s = $('stream');
  if (!s.children.length || $('empty')) return;
  const el = document.createElement('div');
  el.className = 'reset-mark';
  el.id = 'reset-mark-' + sessions;
  el.textContent = 'เริ่มบทสนทนาใหม่ (AI ไม่จำข้อความด้านบน)';
  s.appendChild(el);
  jumpToLatest();
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
      clearAlert();
      // ป้ายปุ่มต้องตรงกับค่าจริงของเซิร์ฟเวอร์ทันที ไม่ต้องรอให้ผู้ใช้กด (P2-13)
      renderMute(!m.tts_enabled);
      renderEcho(!!m.echo_guard);
      if (!voiceFilled) {
        fillVoices(null, m.tts_voice, m.tts_label);
      } else if ($('sel-voice').value && $('sel-voice').value !== m.tts_voice) {
        // เคยเลือกเสียงไว้แล้วแต่เซิร์ฟเวอร์เริ่มมาด้วยค่าจาก .env — สลับให้ตรงกัน
        // (รอ 'setting' ตอบกลับค่อยอัปเดตชิป กันโชว์ค่าที่กำลังจะถูกแทนที่วูบเดียว)
        send({ type: 'tts_voice', value: $('sel-voice').value });
      } else {
        $('chip-tts').textContent = m.tts_label || m.tts_model;
      }
      controls(true);
      $('btn-start').textContent = '● ทำงานอยู่';
      $('btn-start').disabled = true;
      sessions++;
      if (sessions > 1) markMemoryReset();   // session ใหม่ = ความจำเริ่มใหม่ (P2-9)
      // ไม่แสดง path ของ log แล้ว — เซิร์ฟเวอร์ไม่ส่งออกมาให้ client อีกต่อไป (P1-1)
      addLog('พร้อมใช้งาน · บันทึกบทสนทนาไว้ที่เครื่องที่รันเซิร์ฟเวอร์', 'good');
      break;

    case 'phase':
      setOrb(ORB_BY_PHASE[m.value] || 'thinking');
      break;

    case 'calibrated': {
      addLog(`เสียงรบกวน ${m.noise} · เกณฑ์เริ่มอัด ${m.threshold}`);
      $('meter-mark').style.insetInlineStart = (scale(m.threshold) * 100).toFixed(1) + '%';
      break;
    }

    case 'status':
      if (m.text) {
        setStatus(m.text, '');
        // ไอคอนที่ไม่รู้จัก = ระบบกำลังทำอะไรอยู่แน่ ๆ → 'thinking' ไม่ใช่ 'idle'
        setOrb(ORB_BY_ICON[m.icon] || 'thinking');
        if (m.icon === '💭') thinkAt = performance.now();
      }
      break;

    case 'speech':
      if (m.state === 'start') {
        clearAlert();          // กลับมาทำงานได้แล้ว — ไม่ปล่อยให้ error ค้างหน้าจอ
        setOrb('listening');
        setStatus('กำลังฟังคุณพูด...', '');
      }
      break;

    case 'level': {
      const el = $('meter-fill');
      if (el) {          // แผงข้างถูกซ่อนบนจอเล็ก แต่ element ยังอยู่
        el.style.width = (scale(m.rms) * 100).toFixed(1) + '%';
        $('meter').classList.toggle('hot', m.rms > m.threshold);
        $('meter-mark').style.insetInlineStart = (scale(m.threshold) * 100).toFixed(1) + '%';
        $('meter-val').textContent = m.rms.toFixed(3);
      }
      orbLevel(m.rms);          // วงแหวนรอบวงกลมไมค์ (เดิมไม่มีโค้ดขับเลย)
      break;
    }

    case 'begin':  clearAlert(); beginMsg(m.role, m.label); break;

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
      if (m.key === 'mute') renderMute(!!m.on);
      if (m.key === 'echo_guard') renderEcho(!!m.on);
      if (m.key === 'tts_voice') {
        $('sel-voice').value = m.value;
        $('chip-tts').textContent = m.label;
        addLog(`ใช้เสียงพูด: ${m.label}`);
      }
      break;

    case 'cleared':
      $('stream').innerHTML = '';
      turns = 0;
      $('turn-count').textContent = '0 ข้อความ';
      jumpToLatest();
      addLog('ล้างประวัติการสนทนาแล้ว');
      break;

    case 'log':
      addLog(m.text, m.level === 'info' ? '' : m.level);
      // error จาก backend ต้องเห็นได้บนมือถือด้วย ไม่ใช่โผล่แค่ในแผงข้างที่ถูกซ่อน
      if (m.level === 'error') {
        showAlert(m.text);
        setStatus('เกิดข้อผิดพลาด', m.text);
      }
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

// ─────────────────────────────────── ป้ายปุ่มโหมดเสียง (จุดเดียวที่วาดปุ่ม, P2-13)
/* ป้ายต้องบอก "โหมดที่กำลังใช้อยู่" ไม่ใช่โหมดที่จะสลับไป และต้องวาดจากที่เดียว
   ทั้งตอนผู้ใช้กดเองและตอนเซิร์ฟเวอร์แจ้งค่ามา (เดิม event ready/setting แก้แค่
   dataset.on ไม่แตะ textContent ป้ายจึงไม่ตรงกับค่าจริง) */
function renderEcho(guardOn) {
  const btn = $('btn-echo');
  btn.dataset.on = guardOn ? '1' : '0';
  btn.setAttribute('aria-pressed', guardOn ? 'true' : 'false');
  btn.textContent = guardOn ? '🔈 โหมดลำโพง' : '🎧 โหมดหูฟัง';
  btn.title = guardOn
    ? 'โหมดลำโพง: กันไมค์ได้ยินเสียง AI เอง — กดเพื่อสลับไปโหมดหูฟัง'
    : 'โหมดหูฟัง: พูดแทรกไวที่สุด — กดเพื่อสลับไปโหมดลำโพง';
}

function renderMute(muted) {
  const btn = $('btn-mute');
  btn.dataset.on = muted ? '1' : '0';
  btn.setAttribute('aria-pressed', muted ? 'true' : 'false');
  btn.textContent = muted ? '🔇 เสียงปิด' : '🔊 เสียงเปิด';
  btn.title = muted ? 'ตอนนี้ปิดเสียง AI อยู่ — กดเพื่อเปิด'
                    : 'ตอนนี้เปิดเสียง AI อยู่ — กดเพื่อปิด';
}

const echoOn = () => $('btn-echo').dataset.on === '1';
const muted = () => $('btn-mute').dataset.on === '1';

// ───────────────────────────────── ข้อความ error ของไมโครโฟนที่ตรงสาเหตุ (P2-14)
/* เดิมทุก error ได้ข้อความเดียวกันว่า "ตรวจสิทธิ์ไมค์" ทั้งที่สาเหตุที่พบบ่อยที่สุด
   คือเปิดผ่าน http://<ip> ซึ่งเบราว์เซอร์ไม่ให้ navigator.mediaDevices เลย */
const SECURE_HINT = 'ต้องเปิดผ่าน https:// หรือ http://localhost เท่านั้น '
  + 'เบราว์เซอร์ไม่อนุญาตให้ใช้ไมโครโฟนบนหน้าเว็บที่ไม่ปลอดภัย — '
  + 'ถ้าต้องใช้จากมือถือ ให้ทำ SSH port-forward มาที่ localhost หรือเปิด HTTPS';

function micErrorText(err) {
  const name = (err && err.name) || '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'เบราว์เซอร์ปฏิเสธสิทธิ์ไมโครโฟน — กดไอคอนกุญแจ/กล้องบนแถบที่อยู่ '
           + 'แล้วอนุญาตไมโครโฟนสำหรับหน้านี้ จากนั้นลองอีกครั้ง';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'ไม่พบไมโครโฟนในเครื่อง — เสียบไมค์หรือเลือกอุปกรณ์เข้าใน '
           + 'ตั้งค่าเสียงของระบบ แล้วลองอีกครั้ง';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'ไมโครโฟนถูกแอปอื่นใช้อยู่ — ปิดแอปที่ใช้ไมค์ (เช่นโปรแกรมประชุม) '
           + 'แล้วลองอีกครั้ง';
    case 'SecurityError':
      return 'เบราว์เซอร์บล็อกการใช้ไมโครโฟนด้วยเหตุผลด้านความปลอดภัย — ' + SECURE_HINT;
    case 'OverconstrainedError':
      return 'ไมโครโฟนไม่รองรับรูปแบบเสียงที่ขอ — ลองเปลี่ยนอุปกรณ์เข้า';
    default:
      if (!window.isSecureContext) return SECURE_HINT;
      return 'เปิดไมโครโฟนไม่ได้ (' + (name || 'ไม่ทราบสาเหตุ') + ') '
           + ((err && err.message) ? err.message : '');
  }
}

/* ตรวจตอนโหลดหน้าเลย ไม่ต้องรอให้ผู้ใช้กดปุ่มแล้วเจอ error ที่อ่านไม่รู้เรื่อง */
function checkAudioSupport() {
  const canGum = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  if (window.isSecureContext && canGum) return true;
  audioBlocked = window.isSecureContext
    ? 'เบราว์เซอร์นี้ไม่รองรับการอัดเสียงจากหน้าเว็บ (ไม่มี getUserMedia)'
    : SECURE_HINT;
  showAlert(audioBlocked, { sticky: true });
  setStatus('ใช้ไมโครโฟนบนหน้านี้ไม่ได้', audioBlocked);
  for (const id of ['btn-start', 'btn-mic']) {
    const btn = $(id);
    if (btn) {
      btn.disabled = true;
      btn.title = audioBlocked;
    }
  }
  const state = $('mic-state');
  if (state) state.textContent = 'ใช้ไม่ได้บนหน้านี้';
  addLog(audioBlocked, 'error');
  return false;
}

// ───────────────────────────────────────────────────────── เริ่ม/หยุดระบบ
async function start() {
  const btn = $('btn-start');
  if (audioBlocked) {
    showAlert(audioBlocked, { sticky: true });
    return;
  }
  btn.disabled = true;
  btn.textContent = 'กำลังเริ่ม...';
  R.tries = 0;                 // กดเองถือว่าเริ่มนับความพยายามใหม่ (P2-9)
  try {
    await ensureAudio();
  } catch (err) {
    const text = micErrorText(err);
    btn.disabled = false;
    btn.textContent = 'เริ่มระบบ';
    setStatus('เปิดไมโครโฟนไม่ได้', text);
    showAlert(text);
    addLog('ขอสิทธิ์ไมโครโฟนไม่สำเร็จ (' + ((err && err.name) || '?') + '): ' + text,
           'error');
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

$('btn-mute').addEventListener('click', () => {
  const next = !muted();                 // next = ปิดเสียง
  renderMute(next);
  if (next) stopPlayback();
  send({ type: 'mute', on: next });
});

$('btn-echo').addEventListener('click', () => {
  const next = !echoOn();                // next = เปิดโหมดลำโพง (echo guard)
  renderEcho(next);
  send({ type: 'echo_guard', on: next });
  addLog(next ? 'สลับเป็นโหมดลำโพง — กันเสียงลำโพงย้อนเข้าไมค์'
              : 'สลับเป็นโหมดหูฟัง — พูดแทรกไวที่สุด');
});

// ───────────────────────────────────────────────────────── เสียงพูดของ AI
/* ค่า value เป็น "api" หรือ "edge:<ชื่อเสียง>" ตามที่ /api/config ส่งมา
   จำที่เลือกไว้ในเครื่อง เพราะเสียงเป็นเรื่องรสนิยม ไม่ควรต้องเลือกใหม่ทุกครั้ง */
const VOICE_KEY = 'voicelink.voice.v1';
let voiceFilled = false;

function savedVoice() {
  try { return localStorage.getItem(VOICE_KEY); } catch (_) { return null; }
}

function fillVoices(items, current, currentLabel) {
  const sel = $('sel-voice');
  const list = (items && items.length)
    ? items
    : [{ value: current || 'api', label: currentLabel || current || 'เสียงตั้งต้น' }];
  sel.innerHTML = '';
  for (const it of list) {
    const o = document.createElement('option');
    o.value = it.value;
    o.textContent = it.label;
    sel.appendChild(o);
  }
  const want = savedVoice();
  const chosen = list.some((i) => i.value === want) ? want : (current || list[0].value);
  sel.value = chosen;
  // ชิปด้านบนต้องโชว์ตรงกับตัวที่ dropdown เลือกจริง ไม่ใช่ค่าดิบจาก .env เสมอไป
  // (ถ้าเคยเลือกเสียงอื่นจำไว้ใน localStorage ก็ต้องโชว์เสียงนั้น ไม่ใช่ค่าตั้งต้น)
  const match = list.find((i) => i.value === chosen);
  $('chip-tts').textContent = (match && match.label) || currentLabel || chosen;
  voiceFilled = true;
}

$('sel-voice').addEventListener('change', (e) => {
  const value = e.currentTarget.value;
  try { localStorage.setItem(VOICE_KEY, value); } catch (_) { /* ปิด localStorage ไว้ */ }
  stopPlayback();
  if (live) send({ type: 'tts_voice', value });
  else addLog('จำเสียงที่เลือกไว้แล้ว จะใช้ตอนเริ่มระบบ');
});

$('btn-clear').addEventListener('click', () => send({ type: 'clear' }));
$('btn-help').addEventListener('click', () => openTour(0));
$('jump').addEventListener('click', jumpToLatest);
$('alert-close').addEventListener('click', () => clearAlert(true));

/* ผู้ใช้เลื่อนเอง = ตัดสินว่ายังอยากเกาะก้นอยู่ไหม (P2-10) */
$('stream').addEventListener('scroll', () => {
  stickBottom = nearBottom();
  if (stickBottom) showJump(false);
}, { passive: true });

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

/* คู่มือเป็น <dialog> จริง (P3-16): focus trap, background inert, ปิดด้วย Esc
   และการคืนโฟกัสให้ปุ่มที่เปิดโมดัล เป็นหน้าที่ของเบราว์เซอร์ ไม่ต้องเขียนเลียนแบบ
   (เบราว์เซอร์เก่าที่ไม่มี showModal ยังใช้ได้แบบ overlay ธรรมดา) */
function markTourSeen() {
  try { localStorage.setItem(TOUR_KEY, '1'); } catch (_) { /* localStorage ถูกปิด */ }
}

function tourIsOpen() {
  const d = $('overlay');
  return typeof d.showModal === 'function' ? d.open : !d.hidden;
}

function openTour(n) {
  const d = $('overlay');
  d.hidden = false;
  if (typeof d.showModal === 'function' && !d.open) d.showModal();
  showStep(n || 0);
}

function closeTour() {
  const d = $('overlay');
  if (typeof d.close === 'function' && d.open) {
    d.close();          // เหตุการณ์ 'close' ด้านล่างเป็นคนจำว่าดูคู่มือแล้ว
    return;
  }
  d.hidden = true;
  markTourSeen();
}

$('overlay').addEventListener('close', markTourSeen);

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

/* <dialog> ปิดด้วย Esc ให้เองอยู่แล้ว — เส้นทางนี้ไว้เผื่อเบราว์เซอร์ที่ไม่รองรับ */
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (typeof $('overlay').showModal === 'function') return;
  if (tourIsOpen()) closeTour();
});

// ขั้นที่ 1 — ขอสิทธิ์ไมค์
$('btn-mic').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  if (audioBlocked) {
    $('mic-state').textContent = 'ใช้ไม่ได้บนหน้านี้';
    showAlert(audioBlocked, { sticky: true });
    return;
  }
  btn.disabled = true;
  btn.textContent = 'กำลังขอสิทธิ์...';
  try {
    await ensureAudio();
    $('mic-state').textContent = 'เปิดแล้ว · ลองพูดดู';
    btn.textContent = '✓ ไมโครโฟนพร้อม';
  } catch (err) {
    const text = micErrorText(err);
    btn.disabled = false;
    btn.textContent = 'ลองอีกครั้ง';
    $('mic-state').textContent = 'เปิดไมค์ไม่ได้';
    $('mic-why').textContent = text;
    addLog('ขอสิทธิ์ไมโครโฟนไม่สำเร็จ (' + ((err && err.name) || '?') + '): ' + text,
           'error');
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
    // ทดสอบด้วยเสียงที่ผู้ใช้เลือกไว้จริง ๆ ไม่ใช่ค่าตั้งต้นใน .env
    const voice = encodeURIComponent($('sel-voice').value || '');
    const res = await fetch('/api/selftest?voice=' + voice,
                            { method: 'POST', headers: authHeaders() });
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
  renderMute(muted());               // ป้ายปุ่มมาจากฟังก์ชันเดียวตั้งแต่เฟรมแรก
  renderEcho(echoOn());
  paintOrbRing(0);                   // วงแหวนเริ่มที่ศูนย์ (ไม่ใช่ค่าคงในไฟล์ HTML)
  checkAudioSupport();               // แจ้งเรื่อง secure context ก่อนผู้ใช้กดปุ่ม

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
    fillVoices(data.tts_voices, data.tts_voice, data.tts_label);
  } catch (_) {
    addLog('อ่านค่าตั้งจากเซิร์ฟเวอร์ไม่ได้', 'warn');
  }

  let seen = false;
  try { seen = !!localStorage.getItem(TOUR_KEY); } catch (_) { /* localStorage ถูกปิด */ }
  if (!seen) openTour(0);            // <dialog> ที่ยังไม่ open ถูกซ่อนอยู่แล้ว
})();
