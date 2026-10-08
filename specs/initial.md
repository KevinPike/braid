# Local Agent Harness Plan

Source: Claude Docs "Local Agent Harness Plan" (2026-10-08). Milestones are zero-indexed; tick boxes as they land.

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

**Calibrate, don't trust the formula.** For each model, load it twice at two context lengths (for example 4K and 16K), read `size` from `/api/ps` each time, and fit k from the difference. Cache the result per model digest in `~/.harness/calibration.json`.

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

### M3 — Context management

- [ ] Turn on Strands built-in strategies (sliding window, then summarizing) behind a profile setting
- [ ] Custom strategy: summarize with `gemma4:e2b` at 90% of `num_ctx`, keep pinned messages
- [ ] Context Offloader for large tool results (store full output, keep a reference in context)
- [ ] Trim log in SQLite: what was dropped or summarized, when, and why

**Validation:**

1. A 100-turn scripted session never truncates and never exceeds 90% of `num_ctx`.
2. After compaction, the model still answers 4 of 5 recall questions about facts from turn 5.
3. Every trim has a log entry viewable in the TUI.
4. Compaction runs between turns, never mid-generation, and a queued prompt waits for it to finish.

### M4 — Tools and the decision layer

- [ ] Three or four `@tool` functions (read file, list directory, search notes, current time)
- [ ] Strands Shell as the sandboxed shell, bound to one project folder
- [ ] Tool-schema token cost shown in the context panel
- [ ] `decide/tev.py`: `httpx` client for `/v1/systemone` with a 1,500-token input cap
- [ ] Router question (plain chat vs tool task) and compaction-trigger question using `tev1:0.8b`

**Validation:**

1. On a 20-prompt test set, `gemma4:e4b` picks the right tool in at least 16.
2. The tev1 router agrees with hand labels on at least 16 of 20 prompts; every decision and its probabilities are logged.
3. Inputs over the tev1 cap are summarized first, never sent raw.

### M5 — MCP

- [ ] Connect one or two local MCP servers (for example filesystem and a notes server)
- [ ] Per-server schema cost in the context panel; toggle servers on and off live

**Validation:** turning a server off drops its schema tokens from the next call's count.

### M6 — Skills

- [ ] Strands Skills plugin with three local skills
- [ ] Skill selection with `nomic-embed-text` so only matching skills load

**Validation:** context with skills loaded on demand is measurably smaller than with all skills always loaded, with no drop on the M4 test set.

### M7 — Sessions

- [ ] Persist, resume and fork sessions; store the model, `num_ctx` and calibration with each
- [ ] Persist prompt history per session (and globally for the input recall) so Up/Down survives restarts; restore any still-queued prompts on resume
- [ ] Transcript inspector showing each call's exact payload and token count

**Validation:** kill the app mid-session, restart, resume, and the next call's `prompt_eval_count` matches the pre-crash estimate within 5%.

### M8 — Hooks and interventions

- [ ] Approval prompt in the TUI before any shell command or file write
- [ ] tev1 policy check as a first pass before the approval prompt (advisory only)
- [ ] All guard and context events published through hooks, not ad-hoc calls

- [ ] The prompt queue does not advance while an approval prompt is open

**Validation:** a write or shell command never runs without approval, including when tev1 says it is safe; a queued prompt never starts while an approval is pending.

### M9 — Observability and evals

- [ ] OpenTelemetry traces and metrics to a local collector
- [ ] Strands Evals suite over the M4 test set, run against each model in the lineup

**Validation:** one command produces a per-model comparison of accuracy, tokens per second and peak context use.

### M10 — Stretch

- [ ] `claude -p` as an escalation backend or subagent on the Pro plan (subscription limits, no API billing)
- [ ] Subagents with their own context budgets
- [ ] Model routing by task type, driven by tev1
- [ ] KV-cache-friendly prompt ordering (stable prefix first) and its effect on latency

**Validation:** each stretch item has a before-and-after measurement in the M9 eval report.

## Risks and open questions

| Risk | Effect | Mitigation |
|---|---|---|
| Strands Ollama provider may not expose `prompt_eval_count` or per-request `num_ctx` | Breaks M1–M2 accounting | Wrap the provider or write a thin custom provider over the guard client |
| Calibration differs between Ollama versions | Wrong `max_ctx` after upgrades | Key calibration on Ollama version plus model digest; rerun on change |
| tev1 is experimental and has a ~2K context | Wrong routing or policy calls | Advisory only; log every decision; human approval stays the gate |
| Small models fail at multi-step tool use | M4 test set fails | Raise to the 26B MoE for tool tasks; keep tool count under about 8 |
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
