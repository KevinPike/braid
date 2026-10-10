# Local Agent Harness Plan

Source: Claude Docs "Local Agent Harness Plan" (2026-10-08). Milestones are zero-indexed; tick boxes as they land.

## Where we left off (2026-10-08)

**Done:** M0–M4 and M7–M9, all with their validation gates as pytest tests (185 pass, 4 live tests skipped by default; `mypy --strict` clean). **Skipped for now:** M5 (MCP) and M6 (Skills); there are no MCP servers or skills to use yet. **Next:** M11–M15, Workflows (`specs/workflows.md`; planned 2026-10-09), starting with the M11 spike. Goal: learning by using it on hobby projects, with Plans from Claude and execution on local models; the spec says what success means. M10 stays stretch; M5/M6 wait until there is something to connect. Commits `9788338` to `2417067` are M3, M4 and the fixes between them.

**Run it:** `uv run harness [config.toml] [--resume ID | --continue]`; commands are `/model [name]`, `/compact`, `/trims`, `/tools`, `/decisions`, `/sessions`, `/resume <id>`, `/fork [id]`, `/new`, `/inspect [n]`, `/plan <goal>`, `/build [plan-id]`, `/runs [run-id]` (M15) and `/exit`; `ctrl+t` opens the transcript inspector. Model comparison: `uv run --group evals harness-eval`. Live gates: `HARNESS_LIVE=1 uv run pytest tests/test_live_recall.py tests/test_live_tools.py tests/test_live_evals.py`.

**Never run end to end by a person against the full stack.** Everything above is tested with fakes, plus the live gate tests. The first real session should check, in this order:

1. The status bar through startup, a `/model` switch and a compaction (known flicker: `specs/bugs.md`).
2. A long chat until compaction fires: the summarizer (`gemma4:e2b`) loading and unloading, and the "tools" segment of the context panel.
3. The M2 note that Ollama may report a low `prompt_eval_count` on a cached prefix, which would look like Truncation and, now, force a compaction.
4. An approval prompt for a real shell command and a file write (y, n, and Ctrl-C while it is open), and the tev1 policy line it shows.
5. Kill the app mid-session, then `--continue`: history, prompt recall and a paused restored queue.

**Open items carried forward**

- Status bar flicker when a model reloads: `specs/bugs.md`.
- Preflight's `helpers` budget does not yet reserve memory for the summarizer or tev1.
- The shell tool is a working directory, not isolation; since M8 every command waits for approval, and `harness.toml` turns it on (the config default is still off).
- The router runs on every prompt but nothing acts on it until M10; the 20-prompt test set is easy and self-labelled.
- Offloading is only tested at the storage level; M4's tools are the first thing that can produce a large tool result in practice.
- The session `calls` table stores each request's full body, so it grows roughly quadratically over a long session; no pruning yet.
- The tev1 policy question and its 0.5 safe/risky threshold are untested against a live tev1.

## Goal and scope

Build a learning-grade agent harness with a terminal UI that runs local Ollama models on a 24 GB Mac, built on the Strands Harness SDK, with context use and Ollama runtime health visible and guarded at all times.

The target problems are the ones Ollama hides today: silent prompt truncation at `num_ctx`, models spilling from GPU to CPU, unexpected reloads, and not knowing what fills the context window.

**Stack**

- Python 3.12+, managed with `uv`. Written as interview practice: full type hints checked by `mypy --strict`, `asyncio` for the watchdog and streaming, `dataclasses` or Pydantic for API payloads, and `pytest` tests for every validation gate
- Strands Harness SDK (`strands`), not the pre-assembled `strands_harness`, so every layer is visible
- Ollama 0.35+ (required for tev1 decision models), via the Strands Ollama provider plus direct `httpx` calls to `/api/ps`, `/api/show` and `/v1/systemone`
- Textual for the TUI
- SQLite for sessions, metrics and the trim log
- Optional later: `claude -p` as an escalation backend on the Pro plan, never as the main loop

**Hardware constraints**

- 24 GB unified memory. macOS lets Metal use only part of it by default, roughly 16–18 GB; the preflight measures the real limit rather than assuming it.
- One large model (20–26B) loaded at a time, plus at most one small helper.
- KV cache grows linearly with `num_ctx`, so context length is a memory decision, not just a quality one.

**Out of scope:** multi-user serving, cloud deployment, fine-tuning, and API-key billing.

## Model lineup

Use `gemma4:e4b` as the daily driver and tev1 as the harness's decision layer. The 20–26B models are for comparison runs once the runtime guard works.

| Model | Role | Starting `num_ctx` | Milestones |
|---|---|---|---|
| `gemma4:e4b` | Daily driver for the main loop; fast iteration with tool calling | 32K | M0–M5 |
| `gemma4:e2b` | Summarizer for compaction | 16K | M3 |
| `tev1:0.8b` | Decision layer: routing, compaction trigger, tool policy check, done-or-continue | about 2K (fixed by the model) | M4, M8 |
| `tev1:4b` | More accurate decisions, when only a small main model is loaded | about 2K | M4, M8 |
| `gemma4` 26B MoE (26B-A4B) | Stronger agent; about 4B active parameters per token, so it stays fast | 16K | M4–M8 |
| `gpt-oss:20b` | Alternate strong agent and eval judge | 16K | M4, M9 |
| `nomic-embed-text` | Embeddings for skill selection and memory | n/a | M6–M7 |

All `num_ctx` values are starting points; the preflight in M1 replaces them with measured safe maximums.

**Not on 24 GB:** Qwen 3.6 27B, Qwen 3.6 35B-A3B and Gemma 4 31B dense leave almost no room for KV cache once macOS takes its share.

**tev1 notes:** it takes short `state` text and up to 64 questions, returning a choice, a true/false probability or a rubric score. It is not in the Ollama Python library yet, so the harness calls `/v1/systemone` directly. Inputs must stay under about 1,500 tokens, so feed it summaries, never transcripts. It is experimental and never the only check on a risky action.

Verify exact sizes with `ollama show <model>` before committing; they vary by quantization.

## Runtime guard

The guard picks a context length that fits before anything loads, then watches every call for truncation, CPU spill and reloads. It is a standalone module (`guard/`) with no Strands dependency, so it can be tested against a bare Ollama.

### Preflight: picking a safe num_ctx

The memory needed is the model weights plus the KV cache, which grows linearly with context:

```
need = W + ctx * k + O
```

- W = weight size reported by `/api/ps` after load (or `/api/show` before)
- k = KV bytes per token
- O = runtime overhead, about 0.5–1 GB

The theoretical per-token KV cost comes from the architecture fields in `/api/show`:

```
k = 2 * layers * kv_heads * head_dim * bytes_per_element
```

Bytes per element is 2 for f16, about 1 for q8_0. Models with sliding-window layers (Gemma) use less than this formula, so it is an upper bound.

**Calibrate, don't trust the formula.** For each model, load it twice at two context lengths (for example 4K and 16K), read `size` from `/api/ps` each time, and fit k from the difference. Cache the result per model digest in `~/.braid/calibration.json`.

The budget is the Metal limit minus a safety margin and any helper models:

```
max_ctx = floor((gpu_limit - margin - helpers - W - O) / k)
```

- gpu_limit: read `sysctl iogpu.wired_limit_mb`; if it is 0 (the default), fall back to a conservative two-thirds of physical RAM, then confirm with a test load
- margin: 1.5 GB
- Round max_ctx down to a multiple of 1,024 and cap it at the model's trained context

**Preflight checklist (fails loudly, never silently):**

1. Ollama is reachable and at version 0.35 or later
2. Every model in the profile is pulled
3. `OLLAMA_NUM_PARALLEL=1` (parallel slots multiply KV cache); warn if not
4. Optional: `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` to roughly halve KV cache
5. Chosen `num_ctx` fits the budget; otherwise step down to the next smaller model in the profile
6. Load the model, then confirm `size_vram == size` in `/api/ps`

### Watchdog: checks during a session

| Signal | Source | Trigger | Action |
|---|---|---|---|
| CPU spill | `/api/ps`: `size_vram < size` | Any shortfall | Red status; pause the loop; offer a smaller `num_ctx` or model |
| Silent truncation | Per-call `prompt_eval_count` vs harness estimate | Actual more than 5% below estimate | Red banner; log the call; force compaction |
| Context pressure | Harness estimate vs `num_ctx` | 80% (warn), 90% (compact) | Yellow gauge; trigger the compaction strategy |
| Unexpected reload | Per-call `load_duration` | Over 1 s after the first call | Warn; check for `num_ctx` drift or eviction |
| Eviction risk | `/api/ps`: `expires_at` | Under 60 s left | Refresh `keep_alive` |
| Memory pressure | macOS `memory_pressure` / `vm_stat` | Warn or critical level | Yellow or red status |
| Speed drop | `eval_count / eval_duration` | Under 50% of the session median | Warn; usually spill or pressure |

**Rules the guard enforces:**

- One `num_ctx` per model per session. Every request goes through one client that sets it, so it cannot drift and trigger reloads.
- Switching models unloads the old one first with `keep_alive: 0`.
- `/api/ps` is polled every 2 seconds on a background task and never blocks the chat loop.

## TUI layout

The status bar is the guard's output and the context panel is M2's output; both update after every call, and the alert banner appears only when the watchdog fires.

Layout: status bar, chat, context panel, alert banner. The input line sits under the chat; it shows how many prompts are queued, and Up/Down recall earlier prompts (M2).

Status bar colours: green when the model is fully on GPU, yellow on context or memory pressure, red on CPU spill or truncation. Keys open the trim log (M3), model switcher (M1), transcript inspector (M7) and tev1 decision log (M4).

## Milestones

Eleven milestones (M0–M10), run in order. Each one ships something usable and has a validation gate that must pass before the next starts. M1–M3 carry most of the learning, so do not rush past them.

### M0 — Skeleton — DONE (46b4617)

- [x] `uv` project, Textual app with a chat pane and an empty status bar
- [x] Strands `Agent` on the Ollama provider with `gemma4:e4b`, streaming tokens into the chat pane
- [x] Config file (`harness.toml`) for the model profile and paths

**Validation:** a 10-turn chat streams without blocking the UI; Ctrl-C cancels a generation cleanly.

### M1 — Runtime guard — IMPLEMENTED (gate 2 limited by ollama#17251, see notes)

- [x] `guard/ollama_client.py`: the single client every request goes through; pins `num_ctx` and `keep_alive`
- [x] `guard/preflight.py`: the checklist, `/api/show` parsing, Metal-limit read, budget math
- [x] `guard/calibrate.py`: two-load calibration of KV bytes per token, cached per model digest
- [x] `guard/watchdog.py`: `/api/ps` poller plus per-call metric checks from the watchdog table
- [x] Status bar shows model, GPU %, memory, tokens per second and keep-alive

**Validation:**

1. Deliberately set `num_ctx` too high for the 26B model: preflight refuses or steps down, and never loads a model that spills.
2. Force a spill (load a second large model in another terminal): status bar turns red within 4 seconds.
3. Send one request with a different `num_ctx` from outside the client: the reload is flagged.
4. Calibrated k for each model is within 25% of the formula's upper bound, or the gap is explained (sliding-window layers).

**Status notes (2026-10-08):**

- Gate 3 also verified live: a foreign `num_ctx: 2048` request flips `/api/ps` `context_length` and the watchdog raises `num_ctx_drift` (red).
- Gate 4 (explained): the `gemma4` models here load a speculative draft ("gemma4-assistant"), and `/api/ps` reports the draft's footprint instead of the target's ([ollama#17251](https://github.com/ollama/ollama/issues/17251), fix pending in PR #17857). Calibrating from `ps` therefore measures the wrong model, so `Calibrator` refuses (`CalibrationUnreliable`). Budgets use an architecture-aware KV cost instead, which matches the server log exactly for `gemma4:e4b` and `gemma4:12b` (16,384 B/token each, plus a fixed sliding-window cache of 40 MiB and 480 MiB). See ADR 0002.
- Consequence: `size_vram == size` cannot confirm full-GPU for these models. Preflight warns and the watchdog raises a yellow `ps_unreliable` alert until the upstream fix lands. Spill detection for them falls back to the speed-drop check.
- Gate 2 (live spill, red within 4 s) is tested with a fake clock only; for draft-model setups it is not achievable through `/api/ps` today.
- Gate 1 is tested with fakes; the 26B model is not pulled.
- Truncation check takes an `estimate`; M2's ledger supplies it on every call.
- Metal's `recommendedMaxWorkingSetSize` in the Ollama server log is about 18.6 GiB; the budget still uses the conservative two-thirds fallback (16 GiB).

### M2 — Context visibility — IMPLEMENTED (unit-tested with fakes; not yet run against live Ollama)

- [x] Per-call accounting: system prompt, tool schemas, history, tool results, current turn
- [x] Compare the harness's own estimate with Strands' context estimation and with Ollama's `prompt_eval_count`
- [x] Context panel in the TUI: stacked gauge by category plus the last call's numbers
- [x] Truncation detector wired to the red banner
- [x] Chat pane renders basic markdown (headings, bold/italic, inline and fenced code, lists), including while the reply is still streaming
- [x] Prompt history: Up/Down in the input recall earlier prompts (in memory for now; M7 persists it)
- [x] Prompt queue: prompts submitted during a generation wait their turn instead of being dropped, with a count in the TUI. Ctrl-C cancels the current generation and pauses the queue; Enter on an empty input resumes it and Esc clears it

**Validation:**

1. Paste a document larger than `num_ctx`: the truncation banner fires, and the log shows the token gap.
2. Over a 30-turn session, the estimate stays within 5% of `prompt_eval_count`.
3. Category totals add up to the measured total within 5%.
4. Markdown: a streamed reply with an unclosed code fence mid-stream does not crash, and the final render equals rendering the full text at once; plain text with markdown-looking characters (`*`, `_`, `#`) is not mangled.
5. History: after N submissions, Up walks back through them newest first, Down walks forward and ends on the unsent draft; empty and duplicate-consecutive prompts are not recorded.
6. Queue: three prompts submitted during one generation run in order, none lost or reordered, each producing its own reply; the queue holds (does not advance) while a red Alert is active.
7. Ctrl-C mid-generation cancels it and pauses the queue with the remaining prompts still listed; nothing runs until Enter on an empty input resumes (then the next prompt runs) or Esc clears the queue.

**Status notes (2026-10-08):**

- Gates 1–7 each have a pytest test (`tests/test_context.py`, `tests/test_prompts.py`, `tests/test_app.py`). The 5% gates use a fake tokenizer; the estimator learns chars per token from `prompt_eval_count`, only from calls small enough (under 80% of `num_ctx`) that they cannot have been truncated.
- Not verified live: Ollama may report a low `prompt_eval_count` when it reuses a cached prefix, which would look like Truncation. Check on the first real session before trusting the banner.
- The Strands estimate (`Model.count_tokens`, chars/4) is shown in the panel beside ours and Ollama's.
- A red Alert holds the queue until it clears; truncation alerts last 60 s.

### M3 — Context management — IMPLEMENTED (e4b/e2b recall verified live; see notes)

- [x] Turn on Strands built-in strategies (sliding window, then summarizing) behind a profile setting
- [x] Custom strategy: summarize with `gemma4:e2b` at 90% of `num_ctx`, keep pinned messages
- [x] Context Offloader for large tool results (store full output, keep a reference in context)
- [x] Trim log in SQLite: what was dropped or summarized, when, and why

**Validation:**

1. A 100-turn scripted session never truncates and never exceeds 90% of `num_ctx`.
2. After compaction, the model still answers 4 of 5 recall questions about facts from turn 5.
3. Every trim has a log entry viewable in the TUI.
4. Compaction runs between turns, never mid-generation, and a queued prompt waits for it to finish.

**Status notes (2026-10-08):**

- Gates 1, 3 and 4 have pytest tests in `tests/test_compaction.py` with a fake model (prompt tokens measured as chars / 4). Gate 2 is `tests/test_live_recall.py`, skipped unless `HARNESS_LIVE=1`: it passed 5-of-5-or-better (needs 4) with `gemma4:e4b` as its own summarizer. With `gemma4:e2b` as summarizer (`HARNESS_SUMMARIZER=gemma4:e2b`) it also passes. If e2b is not pulled, the configured summarizer falls back to the daily driver and says so in the chat.
- Strategies (`[context] strategy`): `none`, `sliding_window` (Strands' built-in, logged) and `summarize` (default; the custom strategy). Strands' default manager silently drops history after 40 messages, so every agent now names its manager explicitly.
- Strands' `SummarizingConversationManager` is not used: it is sync and runs its own event loop, which cannot share the guard client's httpx connection. The custom strategy reuses Strands' async `generate_summary` and split/pin helpers instead.
- The ceiling holds by projection: compaction fires when last prompt + reply + the largest turn growth seen so far would pass 90% of `num_ctx`. A single pasted prompt bigger than the remaining space still hits the M2 Truncation banner, which also forces a compaction.
- The summarizer runs at its own pinned `num_ctx` (16K), folds long input chunk by chunk, is marked a helper in the Watchdog (no reload or speed alerts, never the status bar's model) and is unloaded after each use. The preflight `helpers` budget does not yet reserve memory for it; revisit if e2b plus the driver gets tight.
- `/trims` lists the Trim log (SQLite at `~/.braid/trims.db`); each new entry also appears in the chat. `/compact` forces a compaction and waits its turn in the queue.
- Offloading uses Strands' `ContextOffloader` over file storage (`~/.braid/offload`) behind a wrapper that logs each offload. No tools exist until M4, so it is covered by a storage-level test only.

### M4 — Tools and the decision layer — IMPLEMENTED (gates 1–2 verified live on a small hand-labelled set)

- [x] Three or four `@tool` functions (read file, list directory, search notes, current time)
- [x] Strands Shell as the sandboxed shell, bound to one project folder
- [x] Tool-schema token cost shown in the context panel
- [x] `decide/tev.py`: `httpx` client for `/v1/systemone` with a 1,500-token input cap
- [x] Router question (plain chat vs tool task) and compaction-trigger question using `tev1:0.8b`

**Validation:**

1. On a 20-prompt test set, `gemma4:e4b` picks the right tool in at least 16.
2. The tev1 router agrees with hand labels on at least 16 of 20 prompts; every decision and its probabilities are logged.
3. Inputs over the tev1 cap are summarized first, never sent raw.

**Status notes (2026-10-08):**

- Gates: 3 is unit-tested (`tests/test_tev.py`, respx). Gates 1 and 2 are in `tests/test_live_tools.py` (`HARNESS_LIVE=1`, hand-labelled set in `tests/data/prompts.json`): `gemma4:e4b` picked the right tool, or none, on 20/20 and the tev1:0.8b router agreed with the labels on 20/20, both with the real tools. The set is small and easy, so treat 20/20 as "the plumbing works", not as a benchmark.
- Tools (`tools/builtin.py`): `read_file`, `list_directory`, `search_notes`, `current_time`, all bound to `[tools] root`; `..`, absolute paths and symlinks out of it are refused. `search_notes` greps `notes/` under the root.
- Shell: Strands' `make_shell` over a sandbox that fixes the working directory to the project folder. That is a working directory, not isolation (a command can `cd ..`), so `[tools] shell` was off by default until M8's approval prompt (ADR 0003).
- Tool-schema cost: it is the "tools" segment of the context panel on every call; `/tools` lists each schema's estimated tokens.
- tev1 (`decide/`): every question and its probabilities go to the Decision log (SQLite at `~/.braid/decisions.db`, `/decisions` in the TUI). Input is capped at 1,500 tokens (3 chars per token, deliberately pessimistic, questions included); over the cap the state is summarized, and if it is still over, the call is refused and logged. tev1 sits outside the guard client because its ~2K window is fixed by the model, so there is no `num_ctx` to pin; it is marked a Watchdog helper like the summarizer.
- The router runs on every prompt in the background and is logged only; nothing acts on it until model routing (M10). The compaction question is asked once the window is 80% full and can only bring a compaction forward; the 90% projection stays the hard rule, and a failed or negative answer changes nothing.

### M5 — MCP — SKIPPED (2026-10-08: no MCP servers to use yet)

- [ ] Connect one or two local MCP servers (for example filesystem and a notes server)
- [ ] Per-server schema cost in the context panel; toggle servers on and off live

**Validation:** turning a server off drops its schema tokens from the next call's count.

### M6 — Skills — SKIPPED (2026-10-08: no skills to use yet)

- [ ] Strands Skills plugin with three local skills
- [ ] Skill selection with `nomic-embed-text` so only matching skills load

**Validation:** context with skills loaded on demand is measurably smaller than with all skills always loaded, with no drop on the M4 test set.

### M7 — Sessions — IMPLEMENTED (gates unit-tested with fakes; see notes)

- [x] Persist, resume and fork sessions; store the model, `num_ctx` and calibration with each
- [x] Persist prompt history per session (and globally for the input recall) so Up/Down survives restarts; restore any still-queued prompts on resume
- [x] Transcript inspector showing each call's exact payload and token count

**Validation:** kill the app mid-session, restart, resume, and the next call's `prompt_eval_count` matches the pre-crash estimate within 5%.

**Status notes (2026-10-08):**

- Gate: `tests/test_sessions.py` runs turns through a real Ledger against a fake Ollama (chars/4), "kills" without any close path, reopens the store and resumes; the next estimate is within 5%, and a cold estimator on the same request is not, so persisting the estimator's learned ratio is what makes it pass.
- Store: `session/store.py`, SQLite at `~/.braid/sessions.db` (sessions, messages, prompts, queue, calls). Strands' `FileSessionManager` is not used: it appends, and compaction rewrites history in place. The message snapshot is rewritten on every `MessageAddedEvent`, after each turn, compaction and model switch.
- Resume drops an unfinished tail (a turn killed mid-reply) and says so; the dropped prompt stays in history. Restored queued prompts come back paused. Resume preflights the stored model and `num_ctx` first, then the profile's.
- Calibration stored per session is the estimator state plus the preflight budget and weights; there is no KV bytes per token to store while calibration is blocked (ADR 0002).
- Transcript inspector: `ctrl+t` or `/inspect [n]`; each call's exact `/api/chat` body (captured by `OllamaClient.on_request`), estimate, Ollama count and error. Summarizer calls are recorded too.

### M8 — Hooks and interventions — IMPLEMENTED (gates unit-tested with fakes; see notes)

- [x] Approval prompt in the TUI before any shell command or file write
- [x] tev1 policy check as a first pass before the approval prompt (advisory only)
- [x] All guard and context events published through hooks, not ad-hoc calls

- [x] The prompt queue does not advance while an approval prompt is open

**Validation:** a write or shell command never runs without approval, including when tev1 says it is safe; a queued prompt never starts while an approval is pending.

**Status notes (2026-10-08):**

- Gates: `tests/test_approval.py` drives a real Strands Agent with a scripted fake model. A denied (or failing) approver leaves no file and no shell side effect even with tev1 at p=0.99 safe; an advisor that fails still asks the approver; a queued prompt never starts while an approval is open; Ctrl-C during an approval denies it and pauses the queue.
- `approval.py`: `ApprovalGate` is an async `BeforeToolCallEvent` hook that sets `cancel_tool` first and clears it only on an explicit approval (fail closed). `shell` and `write_file` are always gated; `[tools] approve` can add tools, never remove those two.
- New `write_file` tool, bound to the root like the read tools. The `search_notes` symlink bug is fixed (`specs/bugs.md`).
- tev1 policy check (`check_policy`, kind `policy` in the Decision log) runs first with a 10 s timeout; its line ("safe p=0.91 (advisory)" or "unavailable") is shown in the prompt and never decides.
- Approval prompt is an inline widget, not a modal, so the input stays visible; it takes focus while open (y approves, n or Esc denies).
- `hooks.py`: watchdog, ledger, Trim log, Decision log, guard client and queue each publish once into a Strands `HookRegistry` (`app.hooks`); every TUI consumer and the session store register there.

### M9 — Observability and evals — IMPLEMENTED (gates unit-tested with fakes; see notes)

- [x] OpenTelemetry traces and metrics to a local collector
- [x] Strands Evals suite over the M4 test set, run against each model in the lineup

**Validation:** one command produces a per-model comparison of accuracy, tokens per second and peak context use.

**Status notes (2026-10-08):**

- Gate: `tests/test_evals.py` runs the command's entrypoint over two fake models and a not-pulled one and checks the table, `.md` and `.json`. Live: `uv run --group evals harness-eval --models gemma4:e4b --limit 3` gave 3/3, about 100 tok/s, peak 1,216 tokens (15% of 8,192) in about 25 s (`tests/test_live_evals.py`).
- Evals (`evals/`): Strands Evals (`strands-agents-evals`, `evals` dependency group) `Case`s over `tests/data/prompts.json` with a deterministic first-tool evaluator; no LLM judge, no cloud credentials. Each model is preflighted at `[evals] num_ctx` through the guard client and unloaded after; models not pulled are listed as skipped. Reports go to `~/.braid/evals/`.
- Telemetry (`telemetry.py`, `[telemetry] enabled = false`): Strands spans plus harness metrics from `app.hooks` (tokens, tokens per second, context share, estimate error, trims, decisions, approvals, alert transitions) over OTLP/HTTP to `localhost:4318`. A missing collector is silent; it costs about 3 s at exit.

### M10 — Stretch

- [ ] `claude -p` as an escalation backend or subagent on the Pro plan (subscription limits, no API billing)
- [ ] Subagents with their own context budgets
- [ ] Model routing by task type, driven by tev1
- [ ] KV-cache-friendly prompt ordering (stable prefix first) and its effect on latency

**Validation:** each stretch item has a before-and-after measurement in the M9 eval report.

### M11 — Spike: real models in the example Execution loop

Spec: `specs/workflows.md` § M11. ADRs 0004, 0005.

- [x] `format` (JSON schema output) on the guard client
- [x] Five kata repos in `~/projects/braid-practice/`, each with a `GOAL.md`
- [x] Plans from `claude -p`, with Behaviors sized for a small model
- [x] `examples/graphs/exec_live.py`: the example loop on a real model under Seatbelt, with run summaries and a report

**Validation:** a no-hints baseline table for `gemma4:e4b` over all five katas (`gemma4:12b` dropped 2026-10-10, see notes), recorded in the status notes (measured, not asserted).

**Status notes (2026-10-09):** validation met with the `gemma4:e4b` table. **2026-10-10:** `gemma4:12b` dropped from the baselines: on this 24 GB Mac it pushed swap past 5 GiB and ran at 3–5 tok/s (see M13's notes).

- 11.1 gate: `tests/guard/test_ollama_client.py` (`format` is sent, `num_ctx` stays pinned, `format` omitted when unset). 11.2–11.4 are throwaway code in `examples/graphs/` (`make_katas.py`, `plan_prompt.md`, `claude_plan.sh`, `exec_live.py`, `report.py`), not under mypy.
- Baseline, `gemma4:e4b`, `--no-hints`, one run per kata, `num_ctx` 16384, Plans from `claude -p` **unreviewed and unedited** (the user's Try it is still to do):

| model | bowling | cron-next | roman-numerals | semver-compare | word-wrap | total Green |
|---|---|---|---|---|---|---|
| gemma4:e4b | 6/6 · 2a · 0r · 0h · 4s · 0.2m | 4/6 · 11a · 2r · 0h · 1s · 1.8m | 6/6 · 1a · 1r · 0h · 5s · 0.2m | 6/6 · 8a · 0r · 0h · 1s · 1.0m | 5/6 · 6a · 0r · 0h · 3s · 1.1m | 27/30 |

  Cell: Behaviors Green/planned, implementation attempts (a), Red rejections (r), hints (h), Behaviors already satisfied by earlier code (s, counted as Green), minutes per Behavior (m). Polish was kept on bowling, roman-numerals and word-wrap and reverted on semver-compare and cron-next.
- **12b was deferred**, then dropped on 2026-10-10 as too memory-heavy for this Mac. `gemma4:e2b` is the optional second row: `uv run --with langgraph python examples/graphs/exec_live.py <kata> --model gemma4:e2b --no-hints`, then `report.py`.
- **Reading the table:** 12 of the 27 Green are "satisfied" (roman-numerals 5, bowling 4, word-wrap 3, semver 1, cron 1), so the Plans' Behaviors are often too close together or the model builds ahead. Real iterative work happened on cron-next (11 attempts, 4/6), semver-compare and word-wrap. One sample per cell, so differences of one Behavior may be noise.
- **Divergences from the spec (decided with the user):** (1) a test that passes at Red marks the Behavior satisfied and moves on instead of routing back to `write_test`; this removes retry loops where earlier code already covers the Behavior, but it weakens "no test passes at Red", so M13 should decide whether to keep it. (2) The model sees only the Behaviors up to the current one and is told to make the smallest change; an earlier run that showed all six overbuilt at Behavior 1 (1/6 Green, 15 Red rejections).
- Run summaries are under each kata's `runs/` (gitignored). Not exercised live: hinted runs, the `clarify` flow, a failing Probe.

### M12 — Sandbox and Workspace

Spec: `specs/workflows.md` § M12. ADRs 0005, 0003.

- [x] `Sandbox` interface with a Seatbelt backend: writes only in the Workspace, no home reads beyond the toolchain, no network, scrubbed env, process-group timeout
- [x] `Workspace`: git worktree per run, dependency sync outside the Sandbox, flagged diff, `apply()` that refuses symlinks and escaping paths
- [x] Chat `shell` runs under the Sandbox too (Approval per command unchanged)

**Validation:** model-style code run in a Workspace cannot read the home folder, write outside the Workspace, reach the network or outlive its timeout; a Workspace diff reaches the project only through `apply()`.

**Status notes (2026-10-09):**

- Built: `sandbox/policy.py` (`SandboxPolicy`, `RunResult`, `Sandbox`, `toolchain_paths`), `sandbox/seatbelt.py` (`SeatbeltSandbox`, `LocalSandbox`, shared `run_process`), `sandbox/workspace.py` (`Workspace`, `Change`); `ProjectSandbox` in `tools/builtin.py` now runs through `SeatbeltSandbox`; `[tools] shell_network` in config and `harness.toml`. Shared fixtures (`fixture_repo`, session-scoped `fixture_venv` with pytest) are in `tests/conftest.py` for M13.
- 12.1 gates, `tests/test_sandbox.py`: (1) `test_denied_read_is_blocked_and_writable_read_works`, (2) `test_writes_only_inside_writable`, (3) `test_network_only_when_the_policy_allows_it`, (4) `test_environment_is_scrubbed_unless_passed`, (5) `test_timeout_kills_the_process_group_grandchild_included`, (6) `test_pytest_from_a_fixture_venv_runs_under_toolchain_paths`. Extra: output trimming and profile escaping.
- 12.2 gates, `tests/test_workspace.py`: (1) `test_new_and_edited_files_reach_the_project_only_through_apply`, (2) `test_a_symlink_that_leaves_the_workspace_makes_apply_raise`, (3) `test_an_edited_conftest_carries_the_runs_on_import_flag`, (4) `test_model_code_in_the_workspace_cannot_write_a_project_git_hook`, (5) `test_remove_leaves_no_worktree`.
- 12.3 gate, `tests/test_tools.py`: `test_an_approved_shell_command_cannot_write_outside_the_root`.
- Divergences and additions: (1) Workspace git commands set `GIT_DIR` and `GIT_WORK_TREE` explicitly instead of trusting the `.git` file inside the Workspace, which model code can rewrite to point at a repository whose config runs commands outside the Sandbox. (2) `.tmp` and `.venv` are left out of `changes()` and `diff()`. (3) New files are made visible by listing `git ls-files --others` and adding them intent-to-add, because `git add -A` with an exclude pathspec fails when the excluded folder is gitignored. (4) `Workspace.create` returns the dirty-tree warning as `ws.warning`; `base` and `run_id` have defaults. (5) The Sandbox `PATH` defaults to `/usr/bin:/bin:/usr/sbin:/sbin` and `HOME` to the first writable path; `ProjectSandbox` passes the host `PATH` (venv first) so installed tools stay findable. (6) `Sandbox.run` has `timeout_s=120` as a default. (7) `ProjectSandbox` raises Strands' `SandboxTimeoutError` on timeout and returns the interleaved output as stdout (stderr empty); it runs to completion and then yields, not live streaming. (8) `.tmp/` added to `.gitignore` because the chat shell's `TMPDIR` is `<root>/.tmp`.
- Open questions: none blocking. Not run live: the TUI Try it (`curl`, `cat ~/.zshrc`).

### M13 — Execution loop (CLI)

Spec: `specs/workflows.md` § M13. ADRs 0004, 0005.

- [x] The Plan schema and a structured model adapter over the guard client
- [x] LangGraph Execution loop: Assumption check, Red/Green per Behavior, Stuck, Sign-off, Polish, Approval of the diff
- [x] `harness-workflow build --plan <path>` and the `[workflow]` config

**Validation:** given a Plan and a git project, the CLI reaches an applied diff or a clean stop, every model call pinned through the guard client and every command in the Sandbox.

**Status notes (2026-10-09):** live baseline rerun for `gemma4:e4b` done (28/30, table below). `gemma4:12b` dropped on 2026-10-10: too memory-heavy for this 24 GB Mac.

- Built: `workflow/plan.py` (`Plan`, `Assumption`), `workflow/model.py` (`WorkflowModel`, `GuardedWorkflowModel`, `WorkflowModelError`), `workflow/prompts.py`, `workflow/execute.py` (`build_execution_graph`, `drive_graph`, `initial_state`, `FileEdit`, `Edits`), `workflow/records.py` (`RunLog`, `Recorder`, `BehaviorStats`, `write_summary`), `workflow/progress.py` (`Progress`, `describe`, `final_report`), `workflow/cli.py` (`harness-workflow build`), `[workflow]` in `config.py` and `harness.toml`, `langgraph` as a dependency (nothing from `langchain-*`). `Workspace` gained `list_files()` (`git ls-files` including new files) and `read_head(path)` so workflow code never runs git itself.
- 13.1 gates, `tests/workflow/test_model.py`: `test_a_valid_reply_parses`, `test_one_invalid_reply_then_a_valid_one_parses_after_exactly_two_requests`, `test_two_invalid_replies_raise`, `test_every_request_carries_format_and_the_pinned_num_ctx`; plus the Plan round trip and min-length check.
- 13.2 gates, `tests/workflow/test_execute.py` (ScriptedModel, LocalSandbox, fixture repo, session venv symlinked in as `.venv`): (1) `test_happy_path_two_behaviors_wrong_once_ends_at_finish_and_applies` (exact node trace), (2) `test_a_test_that_passes_at_red_and_at_head_routes_back_to_write_test_and_three_in_a_row_get_stuck`, (2b) `test_a_test_that_passes_at_red_but_fails_at_head_marks_the_behavior_satisfied_and_leaves_the_workspace_alone`, (3) `test_a_syntax_error_in_the_test_routes_to_write_test_never_write_impl`, (4) `test_green_failing_max_attempts_times_gets_stuck_and_a_hint_resumes_write_impl`, (5) `test_a_failing_probe_interrupts_at_clarify_before_any_file_is_written`, (6) `test_a_sign_off_answer_other_than_yes_or_abort_adds_a_behavior_and_returns_to_write_test`, (7) `test_a_polish_edit_that_breaks_a_test_is_reverted`, (8) `test_a_file_edit_outside_the_workspace_is_not_written_and_counts_as_an_attempt`, (9) `test_answering_no_at_finish_leaves_the_project_unchanged_and_the_workspace_on_disk`. Extras: clarify waive, no-hints skip with Workspace restore, the run log (`test_the_run_log_has_one_rich_line_per_node`: a `write_impl` line has prompt, reply and written files; a `run_green` line has exit code and tail), yes variants.
- 13.3: `tests/workflow/test_cli.py` (parser, a full `run_build` with fakes: progress lines, summary JSON, apply on yes; a no-hints run asks only at `finish`), `tests/workflow/test_progress.py` (fake clock; non-TTY prints nothing live, TTY spinner and ticking seconds, report table), `tests/test_live_workflow.py` (HARNESS_LIVE=1, skipped otherwise; not run).
- **Divergence decided with the user, the Red rule:** M11's "a passing test at Red marks the Behavior satisfied" is kept but made safe by the deterministic `check_satisfied` node. A test that passes at Red is rerun against the HEAD implementation (non-test files changed since HEAD restored, files added outside `tests/` removed, restored again in a `finally`). It fails there: **Satisfied**, recorded and on to `next_behavior`. It passes there too: vacuous, counted as a Red rejection, rewritten with feedback, `stuck` after `max_attempts`. Spec gate 2 is reworded and gate 2b added (`specs/workflows.md` 13.2); **Satisfied** is in `CONTEXT.md`. No config switch.
- Other divergences and additions: (1) the graph takes keyword args `hints`, `recorder` and `log` beyond the spec signature; no-hints behaviour lives in the graph, not the CLI. (2) A Stuck answer of `skip` is also accepted interactively (restores the Workspace as the Behavior found it), as in the spike. (3) Edits are limited to `tests/` while writing a test and to non-`tests/` files while implementing or polishing, as in the spike. (4) A rewritten test starts from the Workspace as the Behavior found it, so a rejected test file cannot linger. (5) Every sandboxed command gets `PYTHONDONTWRITEBYTECODE=1` so a file rewritten within one second cannot run from a stale `.pyc`. (6) `Edits` is constrained to 1-3 files in its JSON schema, so a bad count triggers the model retry. (7) The model retry limit (two requests) means `WorkflowModelError` counts as a failed attempt in the graph, not a crash. (8) The Sign-off interrupt also lists passing tests (`pytest -rA`). (9) `max_polish_rounds` > 1 retries Polish after a revert with the failure fed back; a kept Polish ends the rounds. (10) Summaries go to `<project>/runs/` like the spike (gitignored in the katas); a project without that ignore will see an untracked `runs/`. (11) `[workflow] runs_dir` and `command_timeout_s` added. (12) Node wall time (`node_seconds`) is in every log line; interrupting nodes log only after they resume, so the human's wait is not in it.
- `type: ignore`: none in `workflow/`. LangGraph typing needed `cast`s (`ExecState` from `get_state`, the multi-mode `astream` tuple), a `Protocol` (`_NodeFn`) for wrapped nodes, and a `Literal["__end__"]` constant because `END` is typed `str`.
- Open questions: none blocking. M15 will need `Recorder` in the checkpointed state (it is an external object now) and `RunLog` lines do not replay on resume.
- **Live baseline rerun (2026-10-09), `--no-hints`, `num_ctx` 16384, same unreviewed Claude Plans:**

| model | bowling | cron-next | roman-numerals | semver-compare | word-wrap | total Green |
|---|---|---|---|---|---|---|
| gemma4:e4b (M11 spike) | 6/6 · 2a · 0r · 0h · 4s · 0.2m | 4/6 · 11a · 2r · 0h · 1s · 1.8m | 6/6 · 1a · 1r · 0h · 5s · 0.2m | 6/6 · 8a · 0r · 0h · 1s · 1.0m | 5/6 · 6a · 0r · 0h · 3s · 1.1m | 27/30 |
| gemma4:e4b (M13) | 6/6 · 2a · 0r · 0h · 4s · 2.9m | 6/6 · 7a · 1r · 0h · 1s · 13.3m | 6/6 · 1a · 0r · 0h · 5s · 0.9m | 4/6 · 10a · 0r · 0h · 1s · 7.3m | 6/6 · 7a · 0r · 0h · 2s · 6.1m | 28/30 |
| gemma4:12b (M13) | not run | not run | 0/6, stopped in Behavior 1 after ~95 min | not run | not run | — |

  - e4b: no drop from the spike, so no port bug signalled. 13 Satisfied (spike: 14). Both semver-compare Stuck Behaviors (3 and 6) trace to a model-written Behavior 2 test asserting `compare('1.2.3', '1.2.3-rc1') == 0`, which contradicts Behavior 3; `write_impl` may not edit tests, so it oscillated. Minutes per Behavior are far above the spike's: the Mac was hot (GPU 90–94 °C) and swapping (swap peaked at 5.3 of 6 GiB), so the times are not comparable.
  - 12b: run only on roman-numerals because swap was already above 3 GiB with no model loaded. `write_impl` got empty content twice (`WorkflowModelError`, 1214 s) while Ollama generated thousands of tokens at 3–5 tok/s. Likely cause, not confirmed: the model spends its output on thinking (dropped by `GuardedWorkflowModel`), made worse by memory pressure; the spike sent identical options and never ran 12b, so this is not a port regression. Candidate fixes: send `think: false` and a `num_predict` cap from `[workflow]`. SIGINT did not stop the CLI while a model call was in flight; it needed SIGTERM. On 2026-10-10 the user dropped 12b from the baselines as too big for this Mac; the `think`/`num_predict` fix still matters for any thinking model.
  - Follow-up (2026-10-10): `[workflow] think = false` and `num_predict = 4096` are sent on every model call (`OllamaClient.chat(think=...)`, top-level body key); an empty reply with thinking text now fails with "the model returned only thinking text" in the retry prompt and the `WorkflowModelError`. Ctrl-C cancels the build task (first SIGINT; a second raises `KeyboardInterrupt`): the HTTP stream closes, the Sandbox kills its process group, the Workspace is kept and printed, the summary gets status `cancelled`, exit 130. The live test removes its Workspace in teardown; the two stale worktrees in the roman-numerals kata were removed.
  - Not yet exercised live: a hinted run, the `clarify` flow, a failing Probe.


### M14 — Plan loop (CLI)

Spec: `specs/workflows.md` § M14. ADR 0004.

- [x] Research as model-chosen queries the harness runs in-process, with checked citations
- [x] Plan loop: draft, parallel Critiques (including `spacing`, against Satisfied Behaviors), open questions, human review rounds
- [x] `harness-workflow plan "<goal>"` writes a Plan that M13 consumes unchanged

**Validation:** from a one-line goal the CLI produces a Plan whose every Assumption has a Probe or a checked citation, after at least one human review.

**Status notes (2026-10-10):** built and gated with scripted models; **live check pending** (the user runs it with `gemma4:e4b`: plan the five katas locally, build each with `--no-hints`, and put the table with the Satisfied count per Plan next to Claude's).

- Built: `workflow/research.py` (`ResearchQueries`, `Findings`, `CheckedFinding`, `research()`, `run_query`, `check_citation`), `workflow/plan_loop.py` (`build_plan_graph`, `initial_plan_state`, `PlanState`, `PlanDraft`, `CritiqueReply`, `Questions`), Plan-loop prompts in `workflow/prompts.py` (`DRAFT_SYSTEM`, `SPACING_RULE`, `CRITIQUE_SYSTEMS` incl. `spacing`), `Plan.to_markdown()`, `harness-workflow plan` in `workflow/cli.py` (`run_plan`, shared `with_model` preflight), plan-node lines in `progress.py` (`PLAN_NODES`, `Progress.set_round`, `describe`), `[workflow] plans_dir`, `max_rounds`, `critique_angles`, `critique_prompts` in `config.py` and `harness.toml`. `drive_graph` now serves both loops.
- 14.1 and gates 3-4: `tests/workflow/test_research.py` (read/grep/list limits, outside targets refused and not read, symlink to the outside skipped, citation checks, the two-call flow) and in `tests/workflow/test_plan_loop.py`: (3) `test_a_query_outside_the_project_is_refused_and_nothing_outside_is_read`, (4) `test_a_finding_with_a_bad_citation_is_unverified_and_becomes_an_open_question`.
- 14.2 gates, `tests/workflow/test_plan_loop.py`: (1) `test_one_round_makes_one_call_per_angle_synthesize_once_and_tags_critiques_with_the_round`, (2) `test_research_revise_and_approve_each_take_their_route` and `test_the_last_round_finalizes_without_asking_again_and_notes_the_limit`, (5) `test_an_assumption_without_a_probe_or_verified_evidence_triggers_exactly_one_draft_retry`, (6) `test_the_default_critique_angles_include_spacing_and_its_prompt_carries_every_behavior_in_order`, (7) `test_the_draft_prompt_requires_new_code_per_behavior_and_a_spacing_critique_reaches_synthesize`, (8) `test_the_finalized_plan_loads_and_runs_through_the_execution_graph_to_its_first_interrupt`. Extras: a config-only fifth angle with its own prompt and one run-log line per node, Markdown rendering.
- 14.3, `tests/workflow/test_cli.py`: parser, `test_plan_prints_progress_the_review_and_the_saved_path_and_build_consumes_that_plan` (a saved Plan runs through `run_build` to an applied diff), the TTY round label, model failure (exit 1, nothing saved), and Ctrl-C (stream closed, exit 130, no Plan saved). `tests/workflow/test_progress.py` covers the plan-node lines.
- Divergences and decisions: (1) `draft` asks for a `PlanDraft` (behaviors, assumptions, files, demo, out_of_scope), not a full `Plan`; the loop fills in the goal, `open_questions` and `notes`, so a small model is not shown fields it should leave empty. (2) The human review is always shown, including at round `max_rounds`; there a non-approve answer finalizes with `"stopped at round limit"` and `"unapplied review feedback: <text>"` in `notes`, and `approve` is an ordinary approval. (3) "Verified evidence" on an Assumption means its `path:line` citation passes the same check as findings (file exists, line in range); it need not match a finding. After the one retry, Assumptions still unchecked are dropped from the Plan and become `"Unchecked assumption ..."` open questions. (4) `open_questions` in the saved Plan are the model's up-to-5 questions plus the harness's own (unverified findings, dropped Assumptions), so the list can exceed 5. (5) Model calls go through a one-at-a-time lock, so the five Critiques are parallel in the graph but sequential on the GPU, and the logged latency excludes waiting for a sibling. (6) A Critique or `synthesize` call that fails twice yields no points or questions for that angle instead of ending the run; `research` and `draft` failures end the run with exit 1. (7) Research results shown to the second call are capped at 24 000 characters in total, so they cannot truncate the prompt. (8) `critique_prompts` (angle to system prompt) is an addition so a new angle needs only config; an unknown angle with no prompt gets a generic one. (9) Run logs are `<runs_dir>/plan-<stamp>-<model>.jsonl`; there is no summary JSON for plan runs. (10) Plan ids are `<stamp>-<slug of the goal>`.
- `type: ignore`: none. LangGraph typing needed the `input_schema=CritiqueTask` overload for the `Send` target, `StateGraph[Any]` on `drive_graph` (its state type var is bound to TypedDict-likes), and a `cast` for the final state.
- Open questions: none blocking. M15 will need the Plan-loop state (findings, critiques, feedback) in the checkpoint like Execution's; it is already plain data.
- **Live check commands** (from a kata, with the repo's config so the `[workflow]` block applies):

```sh
cd ~/projects/braid-practice/roman-numerals
H=~/projects/braid/harness.toml
uv run --project ~/projects/braid harness-workflow plan "$(cat GOAL.md)" --project . --model gemma4:e4b --config $H
# answer each review: approve | research: <what to look into> | <revise notes>; the last line names the saved Plan
uv run --project ~/projects/braid harness-workflow build --plan ~/.braid/plans/<id>.json --project . --model gemma4:e4b --no-hints --config $H
```

### M15 — Workflows in the TUI

Spec: `specs/workflows.md` § M15.

- [x] `/plan`, `/build`, `/runs [id]`; interrupts and Approvals as widgets inline in the chat log that hold the Prompt queue and collapse to a record once answered
- [x] Timing feedback in the TUI (live node line, durations, elapsed time in the status bar, per-Behavior table) from the same formatting as the CLI
- [x] Status bar shows every loaded model; the chat model is preflighted again after a run; no empty frame on reload (debug log of polls)
- [x] SQLite checkpoints (including the `Recorder`), resume after Ctrl-C or a killed app
- [x] Workflow hook events and telemetry; `/runs <id>` opens the run log in the inspector

**Validation:** both loops run from the TUI end to end; every human pause and Approval is inline in the chat log and holds the Prompt queue; the TUI shows the CLI's timings; the status bar shows every loaded model and never an empty frame across a swap; a cancelled or killed run resumes from its last checkpoint without repeating model calls.

**Status notes (2026-10-10):** built and gated with fakes (`ScriptedModel`, `LocalSandbox`, `FakeBackend` for `/api/ps`, a fake clock, Textual pilot); **not run live** (302 pass, 5 live tests skipped; `mypy --strict` clean on 91 files).

- Built: `workflow_pane.py` (`WorkflowPane`, `WorkflowDeps`, `ReportingSandbox`; the app's `/plan`, `/build`, `/runs`), `workflow/runs.py` (`RunStore` index table and `open_checkpointer`, in `[workflow] checkpoints_db`, default `~/.braid/workflows.db`), `Recorder.dump/load` and `read_log`, a `recorder` key in both loops' state written after every node, `drive_graph(checkpointer=, resume=, on_resume=)`, `Workspace.attach`, `AnswerPrompt` and an inline `ApprovalPrompt` in `chat.py`, `RunInspector` in `inspector.py`, `GuardState.models` (`ModelStatus`) with `Watchdog.set_active/subscribe_poll`, status bar helpers/total/run clock/history, `debuglog.py`, `WorkflowNodeFinished`/`WorkflowInterrupted` in `hooks.py` and three metrics in `telemetry.py`. `langgraph-checkpoint-sqlite` added. `CONTEXT.md` gained **Checkpoint**.
- Gates, all in `tests/test_app_workflows.py` unless noted: (1) `test_build_runs_to_a_stuck_interrupt_and_a_prompt_submitted_meanwhile_stays_queued_until_the_widget_is_answered`; (2) `test_the_finish_approval_emits_both_approval_events_and_n_leaves_the_project_unchanged` (plus the `y` variant); (3) `test_ctrl_c_during_a_node_then_resume_finishes_without_rerunning_completed_nodes_and_the_table_counts_earlier_behaviors`, with `test_ctrl_c_kills_the_sandbox_process_group_of_the_running_command_and_applies_nothing` and the graph-level `tests/workflow/test_checkpoint.py`; (4) `test_killing_the_app_mid_run_and_resuming_in_a_new_one_continues_from_the_checkpoint`; (5) `test_every_node_update_emits_workflow_node_finished` (plus `tests/test_telemetry.py` for the metrics); (6) `test_a_stuck_interrupt_and_a_chat_shell_approval_are_inline_widgets_that_collapse_to_a_record_and_pytest_shows_inline` (and `tests/test_approval.py`, updated to the inline prompt); (7) `test_timing_the_live_line_updates_in_place_a_finished_node_shows_its_duration_the_clock_runs_and_the_table_matches`; (8) `test_the_status_bar_renders_the_active_model_first_a_helper_after_it_and_a_total_with_both` (plus `tests/test_status_bar.py`, `tests/guard/test_watchdog.py`); (9) `test_swap_a_completed_run_...`, `test_swap_a_cancelled_run_...`, `test_swap_a_failed_run_ends_with_the_chat_model_preflighted_again`; (10) `test_no_empty_frame_across_a_swap_the_loading_state_shows_and_polls_and_renders_are_logged`; (11) `test_runs_with_an_id_opens_the_run_log_in_the_inspector_and_workflow_calls_stay_out_of_the_calls_table` and `test_a_workflow_model_request_is_not_recorded_as_a_session_call`. Extras: `/plan` through review to a saved Plan that `/build` picks up, `/runs` listing, flag parsing.
- Divergences and decisions: (1) a run holds the Prompt queue for its whole length, not only while a widget is open: `/build` is a queued command and runs in the generation worker, so a chat prompt cannot reload the chat model over the workflow model. (2) The run's elapsed time is checkpointed as accumulated active seconds (`Recorder.dump()["elapsed_s"]`), not a start timestamp, so the time a run sat paused is not counted. (3) A node that was cut off reruns on resume and its partial Recorder changes are discarded (the Recorder is restored from the last finished node). (4) Ctrl-C at the `finish` Approval cancels the run (it asks again on resume) instead of denying; chat tool Approvals are still denied by Ctrl-C. `finish` uses the tool name `apply_workspace` and no tev1 advice. (5) TUI runs write no `<project>/runs/*.json` summary; the run log, the checkpoint and the end table are the record. (6) `/plan --resume` works too, since both loops share the code. (7) The answered-prompt record replaces the old hook-driven note ("approved shell: ls" is now "✓ shell: ls? approved"). (8) The "kill" gate exits the app (its cancel handler marks the run paused) and then resets the row to `running` as a real kill would leave it; no process was SIGKILLed. (9) The debug log (`~/.braid/status-debug.log`) is written only while a run is active, from the start of the swap to the chat model's reload. (10) `HarnessApp` gained constructor seams `workflows` (`WorkflowDeps`: model factory, sandbox, backend, clock) and `debug_log`, and a public `note()`. (11) The default workflow model is `[workflow] model`, else the Profile's (`gemma4:e4b`); the `harness.toml` comment no longer suggests `gemma4:12b`.
- `type: ignore`: none added. LangGraph typing needed no new casts beyond `drive_graph`'s existing ones.
- Open questions: (a) The flicker bug stays open: the polls-before-clearing-loading fix and the debug log are covered by fakes only; confirm with `~/.braid/status-debug.log` in a real swap. (b) `RunStore` (sync `sqlite3`) and LangGraph's `aiosqlite` saver share one file; short writes with a 30 s busy timeout, not stress-tested. (c) The five parallel Critiques each get a live line, matched to their result lines in order by node name. (d) Resuming a build whose Workspace is gone (applied, or removed by hand) reports it and stops. (e) Not run live: the swap with real Ollama, the Seatbelt sandbox behind `ReportingSandbox`, a real SIGKILL.
- **Try it (not run):** `uv run harness`, then `/plan add a --version flag`, answer the review (`approve`), `/build` (the most recent Plan) and watch the progress lines, the `$ pytest` lines and the status bar clock; answer a Stuck with a hint; press Ctrl-C mid-node and `/runs` (state `paused`), then `/build --resume <run-id>`; quit braid with the run mid-node (Ctrl-Q), start it again, `/runs` shows it as `killed`, `/build --resume <run-id>`; at `finish` press `n`, then `/runs <run-id>` and step through the nodes in the inspector. Afterwards read `~/.braid/status-debug.log` to see each poll and rendered status around the swaps.

## Risks and open questions

| Risk | Effect | Mitigation |
|---|---|---|
| Strands Ollama provider may not expose `prompt_eval_count` or per-request `num_ctx` | Breaks M1–M2 accounting | Wrap the provider or write a thin custom provider over the guard client |
| Calibration differs between Ollama versions | Wrong `max_ctx` after upgrades | Key calibration on Ollama version plus model digest; rerun on change |
| tev1 is experimental and has a ~2K context | Wrong routing or policy calls | Advisory only; log every decision; human approval stays the gate |
| Small models fail at multi-step tool use | M4 test set fails | Raise to the 26B MoE for tool tasks; keep tool count under about 8 |
| `sandbox-exec` is deprecated by Apple | M12 Sandbox stops working after a macOS update | Behind the `Sandbox` interface; a container backend can replace it (ADR 0005) |
| Small models stall in the Execution loop | Runs end Stuck | Attempt limits and human hints; measure per model in M13's live check; `gemma4:12b` is too memory-heavy on 24 GB (2026-10-10), so smaller Behaviors and hints come first |
| Claude subscription policy for third-party use keeps changing | M10 backend may break | Only use official `claude -p` or Agent SDK; recheck the help article before starting M10 |

**Open questions**

- [ ] TUI only, or also a local browser view (Textual can serve to a browser with `textual serve`)?
- [x] Decided: Python, for interview practice and because Evals is Python-only.
- [ ] Which MCP servers matter most for your daily use?
- [x] Decided: Ctrl-C cancels the current generation and pauses the prompt queue, so a cancelled run never silently starts the next prompt. Resume and clear keys are specified in M2.

## References

- [Strands Agents](https://strandsagents.com) — Harness SDK, context management, plugins, Ollama provider
- [Ollama library: tev1](https://ollama.com/library/tev1)
- [Use the Claude Agent SDK with your Claude plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
- [PromptQuorum: Top open-source models on Ollama](https://www.promptquorum.com/local-llms/top-open-source-models-ollama)
