# TUI demo: five minutes of what braid can do now

Drafted 2026-10-13 on the `openrouter` branch. Terms are from `CONTEXT.md`; one new term is proposed (**Tape**, below). Builds on `specs/openrouter.md` and ADR 0009. Nothing here changes an ADR; two conflicts are surfaced under Risks and Open questions.

Skills used while planning: `domain-modeling` (the Tape term, no new ADR needed), `codebase-design` (the replay seam), `prototype` (UI branch, for the per-call line), `tdd` (each gate is one test at an agreed seam).


**Status (2026-10-13): D1 built** (`src/harness/replay.py`, `tests/demo/test_replay.py`; the `Tape` term is in CONTEXT.md). The offline `--demo choice` (src/harness/demo.py) exists from the Choice work, and the TUI now runs the OpenRouter catalog check before a run. D2-D7 are not started; D8 still needs an ADR 0009 amendment.
## 1. What the viewer walks away knowing

1. **Any model, one flag.** A Workflow runs on Anthropic or on any OpenRouter `vendor/model` id, and the **Catalog** says beforehand what each costs and whether it can return structured output. If it can't, the run is refused before any call.
2. **Models write, code checks.** One model writes the **Plan**, a different, cheaper model builds it **Red** then **Green** in a sandboxed **Workspace**, and pytest decides when it's done (ADR 0004, 0005).
3. **Every call is visible.** For each call you see the served model, the stop reason, the wall time, whether thinking was on, and the cost. A slow call gets a warning while it is still running.
4. **Money is capped and cheap.** The **Spend cap** stops a run at the next node boundary and applies nothing. A kata on DeepSeek costs about $0.005, and all five `claude-plans` katas cost $0.027.
5. **Choices are measured.** `openrouter/auto` records which model it served on each call. The frontier report plots **Hidden tests** pass rate against cost, so "best value for money" is a number.
6. **Honest gap.** The chat agent is still Anthropic-only (ADR 0009).

## 2. Storyboard (8 beats, about 5:00)

Run with `uv run harness --demo demo/tapes/v1` (replay, offline, $0). A `REPLAY ×0.5` badge stays on the status bar the whole time (pace is an open question). The mockups are 100 columns wide.

**Beat 1 (0:00–0:30). Chat on Haiku, with cost visible.** Surface: chat pane and status bar. Type `what does braid do, in one line?`
```
you   what does braid do, in one line?
      Runs Workflows (plan, then build test-first) on any model, with every call costed and capped.
 claude-haiku-5-5 · ctx 3.1k/200k (2%) · in 3.1k · out 41 · cache 2.8k · $0.0011 ·  REPLAY ×0.5
```
Proves: the **Daily driver** on the Anthropic API, and the **Context window** and cost on the status bar.

**Beat 2 (0:30–1:10). The Catalog.** Surface: chat notes. Type `/models *coder* --max-price 0.5`, then `/models --free`.
```
model                                  ctx    in $/M  out $/M  json  tools  note
qwen/qwen3-coder                       262k     0.22     0.95   ✓     ✓    index 54 (aa, 2026-10)
deepseek/deepseek-v4-flash             128k     0.14     0.28   ✓     ✓    index 58 (aa, 2026-10)
3 of 380 structured-output models (461 in the catalog, cached 2h ago) · key credit $4.97 left
```
Proves: the **Catalog**, the price, free and structured-output filters, and the `bench/model-notes.toml` ranking column. Credit is shown as a dollar amount; the key itself never appears.

**Beat 3 (1:10–1:30). Refused before any call.** Type `/build bowling --model liquid/some-chat-model`.
```
/build: liquid/some-chat-model does not support structured outputs; Workflows need JSON (ADR 0009). No call made.
```
Proves: Decision 5 of `specs/openrouter.md`, checked at Preflight.

**Beat 4 (1:30–2:40). Pairing: Claude's Plan, DeepSeek's build.** Surface: the Workflow progress lines in chat. Type `/build bowling --model deepseek/deepseek-v4-flash:nitro --effort low --thinking-sample 0.5`.
```
build bowling-1013 started with deepseek/deepseek-v4-flash:nitro · Plan by claude-sonnet-5-5 · cap $0.10
  check_assumptions ✓ 3 checked  0.4s
  write_test        wrote tests/test_bowling.py  6.2s  (model 6.0s)  $0.0004  end_turn · think 1/1
  run_red           ✓ failed as it should  0.9s
  write_impl        wrote bowling.py  21.7s  (model 21.5s)  $0.0011  end_turn · think 0/1
  ⚠ slow call: deepseek/deepseek-v4-flash:nitro has taken 60s and is still going
  run_green         ✓ all tests pass  1.1s
 …  ·  workflow 0:38  ·  run $0.0049 of $0.10  ·  REPLAY ×0.5
Behavior                        result  attempts  seconds     cost
a gutter game scores 0          green          1     14.0  $0.0009   …
cost $0.0049 · served: deepseek-v4-flash 100%
```
Proves: OpenRouter as a backend, **Plan** provenance (planner and executor are different models), effort, the **Thinking sample** draw for each call, stop reason, wall time, the slow-call warning, and run cost against the **Spend cap**.

**Beat 5 (2:40–3:20). Why did it go there?** Surface: `RunInspector` (`/runs bowling-1013`, then pick an entry).
```
 12 write_impl    │ write_impl  model deepseek/... · served deepseek/deepseek-v4-flash-0731 · stop end_turn
 13   → run_green │ · call 21.5s · thinking_on 0 · $0.0011 · request gen-…
                  │ --- reply --- { "files": [...] }
```
Proves: the **Route record** for each edge (basis: code), the served model, and the thinking summary when one was sampled (ADR 0008).

**Beat 6 (3:20–3:50). The Spend cap.** Type `/build bowling --model deepseek/deepseek-v4-flash:nitro --max-cost 0.002`.
```
  write_impl        wrote bowling.py  19.8s  (model 19.6s)  $0.0012
  spend_cap         ✗ Spend cap: $0.0021 passed $0.00; stopped, nothing applied
```
(The cap shows as `$0.00` because the line formats to 2 places: see Risks.) Proves: the **Spend cap** at a node boundary, and that the **Workspace** diff is never offered.

**Beat 7 (3:50–4:30). Let OpenRouter choose.** Type `/build bowling --model openrouter/auto`.
```
  write_test        wrote tests/test_bowling.py  9.1s  $0.0031  served anthropic/claude-sonnet-5.5
  write_impl        wrote bowling.py  12.4s  $0.0007  served deepseek/deepseek-v4-flash-0731
cost $0.0049 · served (auto, 2026-10-13): claude-sonnet-5.5 40%, deepseek-v4-flash 60%  ~ not repeatable
```
Proves: `openrouter/auto`, the served model for each call, and that an auto run is a dated measurement.

**Beat 8 (4:30–5:00). The frontier, and the gap.** Type `/report --frontier baseline-f1 try-deepseek try-qwen`, then `/model deepseek/deepseek-v4-flash`.
```
| label        | model                  | effort | runs | hidden | cost per kata | frontier |
| baseline-f1  | claude-haiku-5-5       | -      | 5    | 5/5    | $0.0210       | *        |
| try-deepseek | deepseek/...-flash     | low    | 5    | 5/5    | $0.0054       | *        |
/model: the chat agent stays on Anthropic until a Workflow result says to move it (ADR 0009); Workflows take any id.
```
Proves: the benchmark frontier (F12), with hidden pass rate against cost, and states the gap openly.

## 3. Exists vs must be built

| Need | Today | Must build | Where |
|---|---|---|---|
| Chat, status bar with cost | ✓ | REPLAY badge | `src/harness/status.py` |
| `/plan`, `/build`, `/runs`, RunInspector | ✓ (`--model`, `--resume`, `--no-hints` only) | `--effort`, `--thinking-sample`, `--max-cost`, `--backend` flags | `src/harness/workflow_pane.py` (`parse_flags`, `_model_for`) |
| OpenRouter Preflight in the TUI | ✗: `_model_for` calls `make_model` with no Catalog and never runs `prepare_openrouter` | Run Preflight before `_enter`, and pass the Catalog | `workflow_pane.py`, `src/harness/workflow/backend.py` |
| Slow-call warning | CLI only: `warn_slow` writes to stderr, which Textual swallows | `on_slow` → a chat line, wired through `make_model` | `src/harness/workflow/model.py`, `backend.py` |
| Stop reason, call seconds, thinking draws, served model in each node line | In the run log (`records.py`); `describe` shows only `⚠ cut at the token cap` | A suffix in `describe` (shared, so CLI and TUI agree) | `src/harness/workflow/progress.py` |
| Run cost against the cap on the status bar | ✗ | `set_run_cost(spent, cap)` | `status.py`, `workflow_pane.py` |
| Plan provenance (planner) | In progress: uncommitted `PlanAuthor` / `Plan.author` in the working tree on 2026-10-13 | Show `Plan by …` at build start | `src/harness/workflow/plan.py`, `workflow_pane.py` |
| Served-model histogram for auto | In progress: uncommitted `served by:` / `task types:` lines in `final_report` | Percentages, plus the `~` and date marker | `progress.py`, `records.py` |
| `/models`, `/report` | CLI only (`harness-workflow models`, `report --frontier`) | Read-only slash commands reusing the CLI's renderers | `src/harness/app.py` (`COMMANDS`), `src/harness/workflow/cli.py`, `bench.py` (`frontier_table`) |
| `/model` refusal for an OpenRouter id | Misleading text (mentions Ollama) | ADR 0009 wording | `app.py` `_api_model_command` |
| Offline replay | Fakes exist for tests only (`tests/anthropic_fakes.py`, `tests/workflow/fakes.py`) | A Tape, plus replay and recording transports, and `--demo` | new `src/harness/replay.py`; `app.py` `main` |
| Injection points | `HarnessApp(anthropic_client=…)`, `WorkflowDeps.model_factory`, `load_catalog(client=…)`, `key_status(client=…)` | `make_client` / `make_openrouter_client` accept an `http_client` | `src/harness/anthropic_api.py` |

## 4. Recorded replay (offline, $0)

**New term (add to `CONTEXT.md` when D1 lands).** **Tape**: a recorded set of API exchanges (a request fingerprint, the response, its status, its wall time) that replay serves instead of the network. _Avoid_: cassette, fixture, recording (too close to Recorder). No ADR is needed: the change is easy to reverse and sits behind existing seams.

**Seam.** The httpx2 transport. Every model call (chat streaming, Anthropic Workflows, OpenRouter) goes through an `anthropic.AsyncAnthropic`, and the Catalog and key calls go through an `httpx2.AsyncClient`. Faking at the transport fakes all three paths at once, with no change to `AnthropicWorkflowModel`, the Strands agent or the Guard. That gives two adapters, live and replay, so the seam is real.

**Interface** (`src/harness/replay.py`, a deep module):
```python
class Tape:                                   # a directory: exchanges.jsonl + ledger.jsonl + catalog.json + meta.toml
    @classmethod
    def load(cls, path: Path) -> Tape: ...
def replay_transport(tape: Tape, *, pace: float = 0.5, strict: bool = True) -> httpx2.AsyncBaseTransport: ...
def recording_transport(inner: httpx2.AsyncBaseTransport, out: Path) -> httpx2.AsyncBaseTransport: ...
class TapeDrift(Exception): ...               # a request the Tape has no answer for (strict), naming the fingerprint
```
How it works inside:
- **Matching.** The fingerprint is a hash of the normalized body: model, system, messages and schema, minus `session_id`, `metadata` and random ids. Each fingerprint has its own queue, and replay serves that queue in order. This handles parallel **Critiques**, whose arrival order varies. If there is no match: `strict` raises `TapeDrift`, and loose mode serves the next exchange for that model and records the drift.
- **Pacing.** Replay sleeps for the recorded seconds × `pace`, and SSE chunks are spread across that time, so the stream visibly types and the live line ticks. `slow_call_s` is scaled by the same `pace`, so the warning fires at the same point in the run.
- **Recording.** Before anything is written, it drops `Authorization`, `x-api-key` and cookies, and redacts the `sk-…` patterns (`sk-ant-`, `sk-or-`) in bodies. It keeps `request-id` and `usage.cost`.
- **Never reaches the network.** In replay, a request that isn't on the Tape is answered with `TapeDrift` and is never forwarded, even when keys are set in the environment.

**What is faked and what is real.** Faked: every model response, the Catalog (`catalog.json`), `key_status`, and the bench ledger for Beat 8 (`ledger.jsonl`). Real: the Seatbelt **Sandbox**, the git worktree, `uv sync --frozen`, pytest, **Probes**, **Checkpoints** and Route records. The model's files come from the Tape, so the pytest results repeat. `--demo` points `data_dir`, `checkpoints_db` and the workspaces at a temp directory, so `~/.braid` is never touched, and it sets a dummy key that does not match `sk-` so the SDK constructors don't complain.

**Entry.** `uv run harness --demo <tape>` replays. `uv run --env-file .env harness --record <tape>` records a live session (spend approved first). An optional `--autoplay` types the storyboard commands from `meta.toml`, so the demo runs hands-free.

## 5. Build order

Each milestone has a pytest gate that uses fakes (`httpx2.MockTransport`, `LocalSandbox` or Seatbelt, a Textual pilot, a fake clock), with no network or keys. Tests go under `tests/demo/` unless a file is named. Work in order, one at a time.

**D1. Tape and replay transport.** Gate `tests/demo/test_replay.py`: a `FakeApi` exchange is recorded, then replayed byte for byte through `AnthropicWorkflowModel`. Two Critique requests that arrive out of order each get their own reply. An unknown request raises `TapeDrift` and the inner transport sees zero requests. A recorded Tape contains no `sk-` string and no `Authorization` header. A `pace` of 0.5 halves the observed `ApiCall.seconds` within a tolerance.

**D2. Workflow flags and OpenRouter Preflight in the TUI.** `/build` and `/plan` accept `--effort`, `--thinking-sample`, `--max-cost` and `--backend`. The pane runs `prepare_openrouter` before entering a run and passes the Catalog. `on_slow` becomes a chat line. Gate in `tests/test_app_workflows.py`: an unstructured model gets the refusal note, and the fake transport has seen zero completion requests. `--effort low` appears in the request's `output_config`. A 0.5 sample with a seeded rng logs `thinking_on` per call. A fake-clock slow call mounts the `⚠ slow call` line and writes nothing to stderr.

**D3. The per-call line and run cost.** `describe` adds stop reason, `call_seconds`, `think a/b` and `served …` when the served model differs from the requested id. The status bar shows `run $x of $cap`. Gate `tests/workflow/test_progress.py`: fixed records render the exact lines from Beat 4. The CLI and the TUI render the same text for the same record.

**D4. Plan provenance and the auto histogram.** Finish the uncommitted `Plan.author` work (`PlanAuthor`: model, backend, effort) and show it at build start. `final_report` adds the served-model histogram and the `~` marker for auto. Gate: a Plan written by a fake Claude model, built by a fake DeepSeek model, has both in the summary. A run routed to three served models prints the expected percentages.

**D5. `/models`, `/report` and the `/model` refusal.** These are immediate, read-only commands that reuse `cli.py`'s renderers and `bench.frontier_table`. Gate: from a Catalog fixture and a ledger fixture, the pilot output contains the Beat 2 and Beat 8 rows. `/model deepseek/…` prints the ADR 0009 line and leaves the chat model unchanged. No rendered text (chat or `StatusBar.history`) matches `sk-[A-Za-z0-9-]{8,}`.

**D6. `--demo`, `--record`, `--autoplay`.** Gate `tests/demo/test_storyboard.py`: a hand-built Tape for all 8 beats runs under the pilot with `--autoplay` at `pace=0`, at sizes 100×30 and 80×24. Every beat's key line appears, nothing is written outside the temp `data_dir`, and the run finishes in under 60 s.

**D7. Record the real Tape (a live check, not a gate).** Spend must be approved first, at about $0.05: one Haiku chat turn, the DeepSeek `:nitro` build at `low` (about $0.005), one cap-stopped build, and one auto build. The ledger rows for Beat 8 come from existing labels, copied in. Afterwards, `uv run harness --demo demo/tapes/v1` runs D6's assertions against the real Tape, and a script checks that the Tape has no `sk-` string.

**D8 (optional). Chat `/model` on OpenRouter.** Blocked by ADR 0009 ("the chat agent stays on Anthropic until a Workflow result says another model is worth moving it to"). It needs an ADR amendment first, plus a probe of streamed tool use for non-Claude models through the Anthropic-shaped endpoint (unverified). Don't build it for the demo.

**Prototype before D3** (`prototype` skill, UI branch). Build a throwaway Textual app at `prototypes/demo_lines_proto.py` that reads one real run log and switches with a key between 3 layouts for the per-call data: (a) an inline suffix, (b) a dim second line under each node, (c) a right-hand column of calls. It answers "what is still readable at 80 columns?". Keep it on a `proto/demo-lines` branch and fold only the chosen layout into `describe`.

## 6. Risks

- **Terminal size.** Node lines with suffixes reach about 120 columns. Model ids are long (`deepseek/deepseek-v4-flash-0731`). Mitigation: abbreviate the served id to the part after the `/` and drop the `-NNNN` date suffix in the line (the inspector keeps the full id). Wrap the suffix at under 100 columns. D6 asserts at 80×24. Ask for at least 100×30 for the recording.
- **Live API flakiness.** `:nitro` latency varies (20–60 s), auto can route somewhere new every week, free models return 429. That's why the demo is replay by default and recording is a separate step. If presenting live, have the Tape ready as the fallback, set `--max-cost`, and stay off `:free`.
- **Tape drift.** A prompt change in `workflow/prompts.py` changes fingerprints and breaks strict replay. Mitigation: store the harness commit in `meta.toml`, have the D6 gate run on a hand-built Tape that is updated with prompt changes, and re-record the real Tape for each demo (about $0.05).
- **Pacing misleads.** Replayed wall times are scaled. Mitigation: the always-on REPLAY badge, and replay reports the recorded seconds in the lines (see Open questions).
- **Key handling.** Keys come only from the environment (ADR 0009). The recording transport strips and redacts before writing. Replay sets a dummy key that is not `sk-`. `/models` shows credit, not the key. D1 and D5 test that no `sk-` string appears on screen or in the Tape. Never run `--record` with the screen shared until boot is done, and never `cat .env` on camera.
- **Spend cap display.** `describe` formats the cap with `:.2f`, so a $0.002 cap prints as `$0.00`. Fix it in D3 (use 4 places below $0.10).
- **ADR conflict, already in the code.** The TUI `/build --model vendor/model` today skips OpenRouter Preflight and per-reply vetting, while ADR 0009 says Workflows refuse unstructured models before any call. D2 fixes the code; the ADR stays as it is.
- **Sandbox under `--demo`.** Seatbelt is real, so the demo needs macOS and a synced `.venv`. The `uv sync --frozen` in the Workspace must be offline (the cache must be warm). D6 runs it once.

## 7. Open questions (with recommendations)

1. **Replay pace and the times shown.** Show the recorded seconds, or the replayed ones? **Recommend:** sleep scaled, but have replay report the recorded `seconds`, so the lines match reality and the badge explains why it is faster.
2. **Is replay a demo-only flag or a general harness feature?** **Recommend:** general (`--demo`/`--record` on `harness`, and later `harness-workflow --replay`), because it also makes bug reports reproducible.
3. **Does a demo build run in this repo or in a kata project?** **Recommend:** a kata from `bench/katas.toml` (bowling) copied to a temp git repo, so the real project's `uv sync` and diff never come into it.
4. **Approval of the Workspace diff during the demo.** **Recommend:** `--autoplay` answers `n`, so nothing is applied, and the Approval prompt is shown for one second.
5. **Include the Plan loop live?** It's slower and needs human review. **Recommend:** no. Show the saved Claude Plan's provenance line, and mention `/plan` in one sentence.
6. **Chat `/model` to OpenRouter.** **Recommend:** show the refusal (Beat 8) and defer D8 until a Workflow row justifies it, per ADR 0009.
7. **Where Tapes live.** **Recommend:** `demo/tapes/<name>/` in the repo (small, redacted, reviewed), with a pre-commit check that gitleaks scans it.
8. **Hand-edited `bench/model-notes.toml`** ships empty. **Recommend:** fill in two or three rows with source and date before recording, so Beat 2's note column isn't blank.
