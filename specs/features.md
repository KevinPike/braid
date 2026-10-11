# Features after M15

M11–M15 are built (see `specs/initial.md` status notes). These are the next features, planned 2026-10-10 from the live results: e4b gets 28/30 Green with Claude's Plans, but only 11/24 with its own; mixed edits cost attempts; small-model planning does not converge. Terms are from `CONTEXT.md`. Each feature lists its gates; every gate gets a pytest test with fakes, and live numbers are measured, not gated.

**Order:** F1 → F1b → F2 → F3 → F4 → F5 → F6. On 2026-10-10, after the F1 live check, test disputes became F3, `write_change` moved under Later, and benchmarks were added as F2, so that every later feature is measured against a baseline. F1b (the chat agent on Anthropic) follows ADR 0007, which freezes the Ollama path. F1 comes first because a Haiku row turns every later change into a three-way comparison (e4b, Haiku, Claude Plans) at a few cents per run.

---

## F1 — Anthropic API backend for Workflows

Run both loops on Claude through the Anthropic API, so the katas can be run on Claude Haiku 5.5 next to `gemma4:e4b`. Workflows only; the chat agent stays on Ollama.

**Decided in ADR 0006** (amends ADR 0004): the harness is no longer Ollama-only, and anything that needs Claude may use the Anthropic API. Every **Ollama** request still goes through the guard client, so `num_ctx` stays pinned; the Guard is Ollama-specific, and `guard/` never imports `anthropic` (ADR 0001). Data sent to the API leaves the machine, and each run costs money.

**Shape**

- `workflow/model.py` already has the seam: `WorkflowModel.complete(system, prompt, schema) -> T`. Add `AnthropicWorkflowModel(client: anthropic.AsyncAnthropic, model_id: str, *, max_tokens, effort)` next to `GuardedWorkflowModel`. It has the same retry-once-then-`WorkflowModelError` contract.
- Structured output: use `client.messages.parse()` (or `output_config.format` on `messages.create`) with the Pydantic schema; do not parse free text. Check `stop_reason` before reading content: `max_tokens` counts as invalid output (it feeds the one retry), and `refusal` raises `WorkflowModelError` with its category.
- **Haiku 5.5 specifics** (as of 2026-10): model id `claude-haiku-5-5`, $0.10 input and $0.50 output per million tokens (up to 100K-token prompts). Thinking is adaptive and on by default, and effort defaults to `medium`. **Non-default sampling values return a 400, so never send `temperature`.** `GuardedWorkflowModel` sends `temperature` 0.2 today, so do not share that code path. There is no assistant prefill.
- **Prompt caching:** the system prompt and the Plan/context prefix are stable within a run. Mark the prefix with `cache_control` and log `usage.cache_read_input_tokens` to confirm hits. The minimum cacheable prefix depends on the model, so short kata prompts may not cache.
- **Credentials:** use the SDK's default resolution: `ANTHROPIC_API_KEY`, else an `ant auth login` profile. Never put a key in `harness.toml`, the run log or the summary JSON. The Sandbox env scrub (M12) already keeps the key away from model-written code, and a gate proves it (below).
- **Config:** `[workflow] backend = "ollama" | "anthropic"`, and `model` is read per backend. Add a **Spend cap**, `max_cost_usd` per run (default 0.10): the run stops cleanly with status `spend_cap` once the running cost estimate from `usage` passes it.
- **CLI and TUI:** `harness-workflow build --backend anthropic --model claude-haiku-5-5`; `/build --model claude-haiku-5-5` picks the backend from the model id prefix. An API run skips Preflight and the model swap, and the chat model stays loaded.
- **Records:** every JSONL line for a model node gets `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cost_usd` and the API `request_id`. The summary JSON gets totals, `report.py` gets a cost column, and the end table shows cost.
- Dependency: `uv add anthropic`. Check the installed major version; 1.x uses `httpx2`, so do not hand it an `httpx` client.

**Gates** (`tests/workflow/test_anthropic_model.py`, with the SDK's HTTP layer faked: respx or a mock transport, no network):

1. A valid structured reply parses. An invalid reply followed by a valid one parses after exactly two requests, and two invalid replies raise.
2. No request carries `temperature`, `top_p` or `top_k`. The system prompt carries `cache_control`.
3. `stop_reason: "refusal"` raises `WorkflowModelError` with the category. `max_tokens` triggers the retry.
4. The run log and summary carry tokens and cost, and contain no API key even when the key is set in the environment.
5. The Spend cap (`max_cost_usd`) stops a run at the next node boundary, with status `spend_cap`, and nothing is applied.
6. A Sandbox command run during an API-backed build cannot see `ANTHROPIC_API_KEY` (the M12 env scrub, asserted again here).
7. `guard/` does not import `anthropic` (an import-graph test, like the existing one for `strands`).

**Live check:** the five katas with Claude's Plans on `claude-haiku-5-5`, `--no-hints`. Put the row next to M13's e4b row, with cost per kata. Then the five local Plans (M14) on Haiku, which separates "e4b is a weak planner" from "the Plans are bad". Expected cost: cents per kata. Record the real figures.

**Status (2026-10-10): built and gated** (d946339, 5976a1b, b201b93, 9cfefdc; 336 tests).

- **Live check:** `claude-haiku-5-5`, `--no-hints`, default effort (`medium`). The key came from `.env` via `uv run --env-file .env`. No key appeared in any run log or summary. Every call after the first read the system prompt from cache.

| model · Plans | bowling | cron-next | roman-numerals | semver-compare | word-wrap | total Green | cost |
|---|---|---|---|---|---|---|---|
| gemma4:e4b · Claude (M13) | 6/6 · 2a · 0r · 0h · 4s | 6/6 · 7a · 1r · 0h · 1s | 6/6 · 1a · 0r · 0h · 5s | 4/6 · 10a · 0r · 0h · 1s | 6/6 · 7a · 0r · 0h · 2s | 28/30 | local |
| claude-haiku-5-5 · Claude | 6/6 · 3a · 0r · 0h · 3s · $0.004 | 6/6 · 4a · 0r · 0h · 2s · $0.009 | 6/6 · 2a · 0r · 0h · 4s · $0.004 | 6/6 · 5a · 0r · 0h · 1s · $0.008 | 5/6 · 5a · 0r · 0h · 3s · $0.008 | 29/30 | $0.033 |
| gemma4:e4b · local (M14) | 2/6 · 15a · 0r · 0h · 0s | 0/5 · 15a · 1r · 0h · 0s | 4/4 · 1a · 0r · 0h · 3s | 3/5 · 9a · 0r · 0h · 1s | 2/4 · 7a · 1r · 0h · 1s | 11/24 | local |
| claude-haiku-5-5 · local | 4/6 · 9a · 0r · 0h · 1s · $0.012 | 5/5 · 1a · 0r · 0h · 4s · $0.008 | 4/4 · 4a · 0r · 0h · 0s · $0.002 | 5/5 · 2a · 0r · 0h · 3s · $0.007 | 3/4 · 5a · 0r · 0h · 1s · $0.005 | 21/24 | $0.034 |

- **Runtime:** 0.1–0.3 minutes per Behavior on Haiku, against 0.9–13 minutes on e4b. The ten runs cost $0.067 in total.
- **Local Plans are mostly fine.** Haiku gets 21/24 from e4b's own Plans, against e4b's 11/24, so most of M14's gap came from executing them, not from planning.
- **Every Haiku Stuck came from a wrong test, not wrong code.**
  - Word-wrap Behavior 5 (Claude Plan): the test expects `"ij yo"` at width 4, which is 5 characters and breaks the Plan's own width rule.
  - Bowling, local Plan: one test expects 58 for a game the rules score 70; another has an expected value that is an unreadable expression for an incomplete game.
  - Word-wrap Behavior 2 (local Plan): the test's expected output is wrong.

  In each case Haiku said so on all three attempts and would not special-case the implementation. This led to the new F3 (test disputes settled by Plan invariants).
- **Not measured:** effort `low` against `medium`; the Spend cap ($0.10) was never close.

**Later, not in F1:** Haiku as the Plan-loop model (Claude-quality Plans without `claude -p`). The chat agent moved to F1b (ADR 0007). Anthropic Batches for a cheap overnight baseline.

---

## F1b — The chat agent on Anthropic

ADR 0007: Anthropic becomes the default backend and the Ollama path is frozen. The chat agent moves too, so the app runs without Ollama.

- **Model.**
  - `AnthropicChatModel`: a thin Strands `Model` over `anthropic.AsyncAnthropic` (1.x), next to `GuardedOllamaModel` in `agent.py`. Strands' own Anthropic provider requires `anthropic<1.0`, so it can't be used.
  - Reuse Strands' message and tool-spec types; check the `Model` interface against the installed Strands source.
  - Stream with `messages.stream`, map events to Strands `StreamEvent`s, and support tool use.
  - Send no sampling parameters. The system prompt and the stable history prefix get `cache_control`.
- **Config.**
  - `[profile] backend = "anthropic" | "ollama"`, defaulting to `"anthropic"`, with an `anthropic_model` (default `claude-haiku-5-5`).
  - `[workflow] backend` now defaults to `"anthropic"`.
  - A model id starting with `claude-` picks the backend, as in F1.
- **Guard and status bar.**
  - With the anthropic backend, Preflight, Calibration, the model swap and the watchdog's Ollama checks don't run.
  - The status bar shows the model, the tokens used against the real context window (from `usage`), and the session's running cost.
  - The Ledger records API usage in place of `prompt_eval_count`.
- **Compaction and preserved thinking.**
  - Find out, with one cheap live request and from the docs, whether `claude-haiku-5-5` accepts our compaction's rewritten history alongside thinking blocks.
  - If it doesn't, either strip thinking blocks before compacting, or use server-side compaction (`compact-2026-01-12`, beta).
  - Record which one in the status notes. The trigger thresholds move to the model's real window.
- **Ctrl-C** still cancels generation: closing the stream cancels the request.
- **Credentials and safety:** the same as F1. SDK resolution only; no key in config or logs. The Sandbox env scrub covers the chat's tools too.

**Gates** (with `httpx2.MockTransport`, no network):
1. A streamed text reply and a tool-use round trip reach the TUI as the Ollama path does.
2. No request carries sampling parameters; the system prompt carries `cache_control`.
3. With the anthropic backend, no Ollama request is made at startup or during chat (a fake guard client that fails on any call).
4. The status bar shows tokens and cost from `usage`.
5. Compaction produces a history the chosen strategy allows (thinking blocks stripped, or server compaction requested), checked on the request body.
6. Cancelling mid-stream stops the request and leaves the session consistent.

**Live check:** one short chat session with a tool call and a forced compaction on `claude-haiku-5-5`; record cost and cache reads.

**Status (2026-10-10): built and gated** (`tests/test_anthropic_chat.py`, `tests/test_app_anthropic.py`, and a chat-shell key-scrub test in `tests/test_tools.py`).

- **Model.** `AnthropicChatModel` subclasses Strands' `AnthropicModel` only for its pure formatting (Strands messages to API blocks, API stream events to `StreamEvent`s). It builds its own request and sends it on the 1.x client handed in, so the `anthropic<1.0` pin on Strands' provider never matters. Cancelling the consuming task closes the HTTP stream. A refusal ends the turn with a visible note. Prices and context windows moved to `harness/usage.py`, shared with `AnthropicWorkflowModel`.
- **App.** With the anthropic backend there is no `OllamaClient`, Watchdog, Preflight, tev1 or model swap; a gate fails on any of them being built. The status bar shows the model, the last call's tokens against the real window, in/out/cache totals and the session cost. The Ledger and the session's calls table get API usage. `/model` switches between priced Claude models only. Ctrl-C now cuts the history back to the last finished turn, as a resume does.
- **Preserved thinking (live, 2026-10-10).** This account **is enforced by default**. A compacted history (summary + kept turns) that replays a thinking block got a 400 ("bound to a different conversation"), with or without the beta header. The same history with the thinking blocks stripped was accepted. **Chosen strategy: strip.** Compaction and the sliding window remove thinking blocks from the history they rewrite (`strip_thinking`), and the summarizer runs on the chat model through the API. The trigger uses the model's real window (1M), or `[profile] context_window` to compact earlier (Haiku's price goes up 5× past 100K-token prompts).
- **Live check** (headless TUI on `claude-haiku-5-5`, `list_directory` tool call, `/compact`, one more turn): 5 calls, **$0.00075**, 5,378 cache-read tokens. Every call after the first read the system prompt and tools from cache, including the one after compaction. The compaction stripped 2 thinking blocks, and the next turn was accepted and remembered the earlier answers.

---

## F2 — Benchmarks: know whether a change helped

Until now, the comparison tables in the status notes were copied by hand from one run per model × kata. `report.py` keeps only the latest run per cell, and summaries don't record which Plan, harness commit or settings produced them. On the same Plans, e4b moved between 27/30 and 28/30 across reruns, and one kata between 6/6 and 4/6, so a difference of 2 may be noise. A model can now run all five katas for a few cents, so repeats are affordable.

- **Provenance in every summary.** Add `harness_commit` and `harness_dirty`, `plan_path` and `plan_sha256`, `backend`, `model`, `effort`, `max_attempts`, `execution` (graph variant) and `bench` (a run label, or null). This is plain data, so older summaries still load with the fields missing.
- **A manifest** at `bench/katas.toml`, checked into the repo: named sets of `(kata, plan)` pairs.
  - `claude-plans`: the five katas with their `plan.json`.
  - `local-plans`: the M14 Plans.
  - `disputed`: the four F1 Behaviors, with Invariants for F3.
  - The Plans live in the repo under `bench/plans/` so a run can be repeated from them; the kata repos are referenced by path.
- **`harness-workflow bench <set> --model <id> [--repeat N] [--label name]`.**
  - It runs every pair N times, `--no-hints`, answering `no` at finish, and removes each Workspace afterwards.
  - It appends one line per run to a ledger at `~/.braid/bench.jsonl`, holding the summary and its provenance.
  - Every run has its own Spend cap, and the whole bench has `--max-cost`, which stops it cleanly with exit code 3.
- **`report.py --compare <label> <label>`** (or a `bench report` subcommand): per kata and in total, it shows mean Green, the min–max range over repeats, attempts, Satisfied, Stuck, minutes and cost. A difference smaller than the range is reported as "within noise".
- **Comparison only, no hard gate** ("measured, not gated", as before). A drop shows in the comparison; it never fails the test suite.
- **First baseline:** `claude-plans` on `claude-haiku-5-5`, 3 repeats, labelled `baseline-f1`, about $0.10. Every later feature reports its comparison against a baseline in its status notes.

**Gates** (`tests/workflow/test_bench.py`, with fakes):
1. A summary carries the provenance fields, and the Plan hash changes when the Plan does.
2. `bench` runs every pair × N, appends one ledger line per run, and leaves no Workspace behind.
3. The bench-wide `--max-cost` stops cleanly with exit code 3, keeping the runs already written.
4. The comparison computes means and ranges from ledger lines, and labels a difference within the range as noise.
5. Old summaries without provenance still load in the report.

**Status (2026-10-10): built and gated** (48d1287; `tests/workflow/test_bench.py`).

- **Provenance.** `build_once` (behind `run_build`) writes the fields into every summary. `plan_sha256` hashes the Plan's canonical JSON, so reformatting a file does not change it but any edit does. `execution` is the graph variant (`EXECUTION` plus F3's switches).
- **Manifest.** `bench/katas.toml` has `claude-plans`, `local-plans` (the M14 Plans, with the 4-Behavior word-wrap Plan the F1 row used), and F3's `disputed` and `disputed-control`. Pairs may carry a `name`, needed when a set holds a kata twice. The ledger and report key on it.
- **Bench.** A fresh model per run, so usage never leaks between runs (the run's usage cursor also starts at the model's current call count now). Each run's Spend cap is the smaller of `max_cost_usd` and what is left of `--max-cost` (default $1.00). The Ollama backend is refused (ADR 0007). Each Workspace is removed after its ledger line is written.
- **Report.** `harness-workflow report` lists labels, shows one, or `--compare A B`. A difference smaller than the wider of the two min–max ranges is "within noise" (as the spec says; equal is not noise). With one run on either side, the verdict says the noise is unknown. A repeat that is missing a kata is left out of the total. `--summaries <practice-dir>` replaces `examples/graphs/report.py`, which now calls it.
- **Baseline `baseline-f1`** (`claude-plans`, `claude-haiku-5-5`, 3 repeats, `--no-hints`): **29.7/30 [29–30]**, 19.7 attempts, 12.3 Satisfied, 0.3 Stuck, 4.5 minutes and **$0.033 per repeat** ($0.1003 for 15 runs). The only miss was word-wrap Behavior 5 in one repeat, the F1 dispute again. Runs after the first few record `harness_dirty: true`: I was editing F3 while the bench ran, but the process had already imported the committed 48d1287 code.

| kata | baseline-f1 (3 repeats) |
|---|---|
| bowling | 6.0/6 [6–6] · 3.0a · 3.0s · 0 stuck · 0.7m · $0.0043 |
| cron-next | 6.0/6 [6–6] · 4.0a · 2.0s · 0 stuck · 1.0m · $0.0086 |
| roman-numerals | 6.0/6 [6–6] · 2.7a · 3.3s · 0 stuck · 0.6m · $0.0044 |
| semver-compare | 6.0/6 [6–6] · 5.0a · 1.0s · 0 stuck · 0.9m · $0.0077 |
| word-wrap | 5.7/6 [5–6] · 5.0a · 3.0s · 0.3 stuck · 1.2m · $0.0084 |
| total | 29.7/30 [29–30] · 19.7a · 12.3s · 0.3 stuck · 4.5m · $0.0334 |

---

## F3 — Test disputes, settled by Plan invariants

The F1 live check: all four Haiku Stucks were tests whose expected value contradicts the Plan. For example, word-wrap expects `"ij yo"` at width 4, and a bowling test expects 58 for a game the rules score 70. Haiku said the test was wrong on every attempt, spent three attempts, and the run went to Stuck without saying why. An invalid test that reaches implementation should be caught and sent back to `write_test` with the reason. A model's claim must never decide the route (ADR 0004), so a deterministic check decides it, against invariants the Plan states.

- **Invariants in the Plan.**
  - `Plan.invariants: list[Invariant]`, where `Invariant(text, check)`. `check` is a Python expression over one call of the function under test: `fn` (its name), `args`, `kwargs` and `result`. Examples:
    - `all(len(line) <= args[1] for line in result.split("\n"))` (word-wrap)
    - `0 <= result <= 300` (bowling)
  - The Plan loop's `draft` writes them, from the goal's rules. F4's Plan checks add a check that a Plan with rules in its goal has at least one invariant.
  - Hand-written Plans (`plan.json`) can add them by hand.
- **Extracting the asserted examples.** A deterministic helper parses a test file with `ast` and finds `assert fn(<literals>) == <literal>`, both ways round, including inside `pytest.mark.parametrize` tables when the values are literals. It returns `(test name, fn, args, kwargs, expected)` for each. Asserts it can't evaluate with `ast.literal_eval` are skipped and counted, never guessed.
- **`check_invariants`**, a deterministic node.
  - It evaluates every invariant on every extracted example, with `result = expected`, in the Sandbox.
  - A violation returns `{test, example, invariant, value}`.
  - It runs at two points:
    1. **After `run_red`**, on the new test, before implementation sees it. A violation goes back to `write_test` with the invariant and the example named, and counts as a Red rejection.
    2. **On a dispute.** `write_impl`'s reply schema gains an optional `dispute: {test, reason}`. A dispute does not route by itself: it triggers `check_invariants` on the disputed test.
       - A violation goes back to `write_test` for that Behavior with the implementer's reason and the failed invariant. The implementation attempt is not counted.
       - No violation means the dispute is not upheld, and the attempt counts as usual. The dispute reason goes into the next prompt and into the run log.
- **Stuck shows the dispute.** A Behavior that ends Stuck with disputes on record shows the last reason and any invariant that was checked. This lets the human fix the test or add the missing invariant, then resume.
- **Earlier Behaviors' tests:** a dispute naming an earlier Behavior's test goes through the same check. A violation sends the work back to that Behavior's `write_test`, then reruns its Red/Green before continuing. This is the semver-compare case from M13. Otherwise it goes to Stuck as now.
- **Prototype first** in `examples/graphs/` with a scripted model. Then replay the four disputed Behaviors from the F1 live check, using the Plans with invariants added by hand, on Haiku and on e4b. Port it into `workflow/execute.py` once a corrected test reaches Green in those replays.

**Gates** (`tests/workflow/test_execute.py` and a new `tests/workflow/test_invariants.py`):
1. The extractor finds plain and parametrized literal asserts, in both orders, and skips non-literal ones without guessing.
2. A new test whose expected value breaks an invariant goes back to `write_test` after `run_red` with the invariant named, and implementation never sees it.
3. A dispute on a test that breaks an invariant goes back to `write_test` with the reason and does not use an implementation attempt. The corrected test then reaches Green.
4. A dispute on a test that breaks no invariant is not upheld: the attempt counts, and the reason is in the next prompt and the log.
5. A Behavior that ends Stuck after disputes shows the dispute reason in the Stuck interrupt.
6. An invariant expression that raises counts as "unchecked", not as a violation, and is reported.

**Status (2026-10-10): built and gated** (3a271b4, 62527a3; `tests/workflow/test_invariants.py`, the F3 section of `tests/workflow/test_execute.py`).

- **Extractor** (`workflow/invariants.py`). It handles the shapes in spec, plus one deliberate extension. Three of the four disputed tests had values that `ast.literal_eval` rejects: `3 + 7 + ...`, `[3, 4] + [0] * 18`, and `rolls = [...]` followed by `assert score(rolls) == 17`. So the evaluator also folds `+ - *` over values it already knows, and follows straight-line `name = <value>` assignments in the test. This is still exact; calls, attributes and comprehensions are skipped and counted. A single parametrize argname takes each row whole, as pytest does. `Invariant` gained an optional `fn`, which limits a rule to calls of that function.
- **Checking.** The checker script and its JSON input go into the Workspace's `.tmp/`. It runs in the Sandbox with the test command's Python and `-I`, under a restricted builtins table, and reads values back from `repr` with `ast.literal_eval`. If the checker prints no result, that is an error, never a pass.
- **Graph.** `check_invariants` runs after a `red` verdict, and on a corrected earlier test. `write_impl` replies `ImplEdits`: edits may be empty when there is a `dispute`. `check_dispute` matches the disputed test by function name, else by file. An upheld Dispute on an earlier Behavior's test sets `revisit`: that Behavior's `write_test` rewrites the test without restoring the snapshot, then `check_invariants` runs, then `resume_behavior` returns to the disputing Behavior's `run_green`. A Dispute that is not upheld still runs the suite, so the next prompt has its output as well as the note. Stats gain `disputes` and `disputes_upheld`. Stuck shows the last Dispute in the CLI and the TUI.
- **Switches** (`[workflow]`, recorded in the summary's `execution`): `show_invariants` (the rules appear in the models' context) and `check_tests_at_red`. With the Red check on, an upheld Dispute is reachable only when the Red check was bypassed, so the bench flags `--no-red-check` and `--hide-rules` exist to measure each mechanism on its own. The gates for upheld Disputes run with the Red check off.
- **Order of work.** The prototype (`examples/graphs/invariants_proto.py`, no model) showed that hand-written Invariants reject a test in **all four** disputed Behaviors, and no correct example beside them. I ported it before the replays rather than after; the replays below are the evidence that a corrected test reaches Green.
- **Open question decided by the data, not here:** whether to send a Dispute that no Invariant covers back to `write_test` anyway. In the control, Haiku's Dispute was correct every time a Behavior went Stuck (6/6). That argues for the ADR 0004 amendment, or for an arbiter, but it is the user's call.

**Replays** (`claude-haiku-5-5`, 3 repeats, `--no-hints`; the `disputed` set's three Plans, which cover the four F1 Behaviors):

| label | what differs | Green | Stuck | attempts | cost / repeat |
|---|---|---|---|---|---|
| `f3-control` | the same Plans with no Invariants | 14.0/16 [13–15] | 2.0 | 12.3 | $0.0178 |
| `f3-dispute-only` | Invariants, rules shown, `--no-red-check` | 15.3/16 [15–16] | 0.7 | 8.0 | $0.0163 |
| `f3-disputed` | Invariants, rules shown, Red check on (the default) | **16.0/16 [16–16]** | **0** | 6.7 | $0.0150 |

- **`f3-disputed` against the control: +2.0, better** (the ranges do not overlap). Bowling went from 4.7 to 6.0 and word-wrap/claude from 5.3 to 6.0, with half the attempts and less cost. Two of the exact F1 bad tests came back and were rejected at Red before implementation saw them: the first-frame strike `== 17` and word-wrap `"ij yo"`. Both were rewritten and reached Green. In the other seven runs Haiku wrote tests that kept the rules on its first try, which suggests that showing the rules matters as much as checking them.
- **Dispute-only:** the word-wrap B5 Dispute was upheld 3/3, and each time the corrected test reached Green. The two bowling Stucks were bad tests that the hand-written Invariants do not cover: a strike in a middle frame (the rule covers only the first frame), and `13` where `16` is right (above the pin total, so `result >= sum` holds). Stuck showed Haiku's correct reason both times.
- **Control:** six Stucks in nine runs, all with a correct Dispute that nothing could uphold, just as in F1.
- **No regression on Plans without Invariants.** F3 adds the Dispute instruction to every `write_impl` prompt. On the F3 harness, `f3-claude-plans` (3 repeats) is **the same** as `baseline-f1` on every kata: 29.7/30 [29–30], 19.7 attempts, $0.0327 against $0.0334 per repeat. Its one miss is still word-wrap B5, because `claude-plans` has no Invariants. Next: add Invariants to the `claude-plans` Plans and rebaseline.
- **Live spend for F1b–F3:** about $0.35 in total. That is the bench ledger's $0.3456 over 51 runs, plus the F1b probe and chat check ($0.0009).

## F4 — Mixed edits and Plan checks

**Mixed edits** (still needed for Polish and the split graph). When a reply mixes allowed and refused files, write the allowed files, **still run the check**, and put the refusal in the next prompt. Gate: a `write_impl` reply with the implementation plus a test file runs `run_green` on the implementation and does not use up a second attempt for the refusal alone.

**Plan checks after `draft`.** These are deterministic checks, and a failure goes back to `draft` with the reason (at most once, as with Assumptions):
- Every path in `files` exists in the project, or is marked new (a new `new_files` list, or a `(new)` suffix).
- `demo` is a valid command: for `python -c`, the code argument compiles.
- **A Probe must be able to fail.** Run it once as written (it should pass), then once against a Workspace whose module is emptied back to its docstring. A Probe that passes both times proves nothing; flag it, and after the one retry drop the Assumption and make it an open question.

**Review convergence.** `synthesize` sees the questions already asked and answered, and returns only new ones. A round with no new questions offers `approve` as the default. Gate: answered questions do not come back in the next round.

**Gates** (`tests/workflow/test_plan_loop.py`, extended): a Plan with a missing file, an invalid demo or a probe that cannot fail gets exactly one draft retry naming the problem; answered questions are not asked again.

---

## F5 — Time travel: fork a run from a checkpoint

LangGraph keeps every checkpoint (M15), so a stuck run can be replayed from just before its bad node with a different prompt, hint or model.

- `/runs <id>` lists the run's checkpoints (node, Behavior, attempt).
- `/runs <id> --fork <checkpoint>` creates a new run id from that checkpoint's state, in a fresh Workspace: a new worktree with the checkpointed file contents written in. It optionally takes `--model` and `--hint`. The original run is never changed.
- The CLI gets the same command: `harness-workflow fork <run-id> <checkpoint> [--model] [--hint]`.

**Gates:** a fork replays from its checkpoint without repeating earlier model calls (count `ScriptedModel` calls); the original run's checkpoint and Workspace are unchanged; a fork with `--model` uses that model for its next node only from that point.

**Try it:** fork the semver-compare Behavior 3 run from M13 on Haiku and see whether it gets past the contradictory test.

---

## F6 — Graphs defined in YAML, as skills

Small Workflows written as YAML files and run with `/run <name> [args]`. This revives M6 (Skills) as small checked graphs, not prompt files.

- **Registry:** `workflow/registry.py` maps a name to a builder, an argument parser and a description. `/plan` and `/build` register there, and `/run` lists the entries.
- **YAML:** files in `skills/` (in the project, then `~/.braid/skills/`). There are three node kinds:
  - `run`: a Sandbox command; its result is the exit code and output.
  - `model`: a system prompt, a prompt template, a schema from a registry, and an optional `writes` list of allowed paths.
  - `ask`: an interrupt.

  An optional `python:` escape hatch names a registered function. Edges carry conditions only on a `run` exit code, an `ask` answer or attempt counters, so a route cannot depend on what a model claims (ADR 0004).
- **The loader validates at load time:** unknown node kinds, unreachable nodes, routes without a fallback, `{node.field}` references to fields no earlier node produces, and `writes` outside the project. It compiles into the same `StateGraph`, so checkpoints, the run log, progress and resume all apply unchanged.
- **First skills:** `changelog` (git log → draft entry → review → write `CHANGELOG.md`) and `regex` (examples → model regex → run against examples → retry).
- **Prototype first** in `examples/graphs/yaml_graph.py` with a scripted model, to check that the format reads well, before it goes into `workflow/`.

**Gates:** each load-time check rejects a bad file with a message naming the node; a valid file compiles and runs to its first interrupt with a scripted model; a condition on a model field is rejected at load time unless the edge declares `basis: model` and a bound (ADR 0008).

---

# Features after F6: multi-agent Workflows and evals

Planned 2026-10-11. On the API, separate roles, separate models per role and larger repeat counts cost cents. On the frozen Ollama path, each meant a model swap and minutes per call. ADR 0008 amends ADR 0004: a model's claim may route a Workflow when the route is traced and bounded.

**Order:** F7 → F8 → F9 → F10 → F11 → F12 → F13. F7 comes first because every later feature is measured on katas that can't have been memorized, graded by hidden tests. F9 comes before F10 because routing on a claim requires the trace. F7–F9 don't depend on F4–F6; whether they go ahead of F4–F6 is the user's call.

**Context separation today.** Every model call is already a single-turn request with no conversation history: `WorkflowModel.complete(system, prompt, schema)`. What the roles share is the **system prompt** (`prompts.SYSTEM` for both `write_test` and `write_impl`), the **model**, and the **view**: `context()` gives both roles the full contents of every Plan file and every edited file, so the test writer sees the implementation (white-box). F11 separates role, model and view.

---

## F7 — Exotic katas, graded by hidden tests

The five katas are classics that are surely in training data, and the only grade is the model's own tests going Green. An Opus subagent wrote six invented katas under `bench/exotic/` (2 easy, 2 medium, 2 hard), each with `GOAL.md`, a reference implementation, a **hidden test suite**, five mutants that each fail at least one hidden test, and a Plan with Invariants. `bench/exotic/verify.py` checks all of this.

- **Repos:** `make_katas.py` also creates the exotic kata repos, from each kata's `GOAL.md` and `summary.txt`. Hidden tests and reference code are never copied into a kata repo or a Workspace.
- **Manifest:** a set `exotic` in `bench/katas.toml`. A pair may carry `hidden = "exotic/<kata>/hidden"`.
- **Grading:** after `finish` and before the Workspace is removed, copy the final non-test files and the hidden tests into a scratch copy, then run pytest there in the Sandbox. The summary and ledger get `hidden_passed` and `hidden_total`. Grading is code, not a model.
- **Test strength:** run the model-written tests against each mutant, giving `mutants_killed` out of 5. This measures the tests, not the code.
- **Report:** columns for hidden pass rate, the **self/hidden gap** (Green on its own tests but failing hidden ones) and mutants killed.
- **Canary:** every hidden file carries `# braid-hidden-canary: <id>`.

**Gates** (`tests/workflow/test_hidden_grade.py`, fakes):
1. `verify.py` passes: the reference passes every hidden test, each mutant fails one or more, every Invariant holds on every hidden example, and every `plan.json` loads.
2. No prompt in any run log contains a hidden canary (a scripted run on an exotic kata).
3. Grading runs in the Sandbox, and the project's own tests directory is not changed by it.
4. The summary and ledger carry `hidden_passed`, `hidden_total` and `mutants_killed`, and older ledger lines still load.

**Live check:** `exotic` on `claude-haiku-5-5`, 3 repeats, labelled `baseline-exotic`. Report the self/hidden gap next to `baseline-f1`.

**Status (2026-10-11): built and gated** (`tests/workflow/test_hidden_grade.py`, 11 tests).

- **Repos.** `make_katas.py` reads only `GOAL.md` and `summary.txt` from each `bench/exotic/<kata>/` (`exotic_katas()`) and writes GOAL.md as is (it already has its heading). `make(..., sync=False)` skips `uv sync`, for tests.
- **Manifest.** `[sets.exotic]` holds the six katas with their own `plan.json` and `hidden = "exotic/<kata>/hidden"`. A `Pair` carries `hidden`; the mutants and reference are found beside it.
- **Grading** (`bench.grade`, called by `run_bench` after the run and before the Workspace is removed; a cancelled run is not graded). Each pytest run gets a fresh scratch copy in the system temp dir, outside the Workspace and the project, with the Workspace's `.venv` linked in so the run's own `test_command` works. The Sandbox may write only in the scratch copy, and the copy is removed afterwards. The grade is added to the summary file (`records.amend_summary`) and the ledger line, and printed on the bench line. Where the spec was silent:
  - A test file is anything under `tests/` or `test/`, `test_*.py`, `*_test.py` or `conftest.py`; every other final file goes in with the hidden suite.
  - `hidden_total` is what the hidden suite collects against the reference, in its own scratch copy. Code that does not import scores 0/46, not 0/0.
  - A mutant replaces `<module>.py` among all the final files, tests included. It is killed when the model-written tests exit 1 on it (as in verify.py). A suite that fails on the run's own code kills none. `mutants_total` is recorded too.
- **Report.** When a label has graded runs, `report <label>` adds three columns: hidden pass rate, self/hidden gap and mutants killed. `--compare` adds one "hidden · gap · mutants" column per label. The gap follows CONTEXT.md: the count of graded runs Green on every Behavior that still fail a hidden test, shown as "1/3 runs". The total row counts the gap kata by kata, and it is graded only for repeats in which every kata was graded. Tables for older labels are unchanged, and older ledger lines load.
- **Seatbelt.** A manual offline probe graded a Workspace with a real `.venv` under `SeatbeltSandbox` with the default `.venv/bin/python` test command: 22/46 hidden, 1/5 mutants, the same as the tests' `LocalSandbox`.
- **Live check: pending user approval.** First run `make_katas.py`, which creates the exotic repos with `uv sync`; then `harness-workflow bench exotic --model claude-haiku-5-5 --repeat 3 --label baseline-exotic`.

---

## F8 — Fault injection for the API path

Nothing in `workflow/` catches `anthropic.APIError` today. A request that still fails after the SDK's own retries escapes the node. The SDK retries 408, 409, 429, 5xx (including 529 overloaded) and connection errors, twice by default. Test the harness's behavior past that point with fakes, at no cost.

- **`FaultyTransport`** in `tests/anthropic_fakes.py`: an `httpx2.MockTransport` that serves a script of responses. These include 429 with `retry-after`, 529, 500, a dropped connection, a stream cut off before `message_stop`, a body that isn't JSON, `stop_reason: max_tokens`, and `refusal`.
- **Workflow policy:** an API error that survives the SDK's retries ends the run with status `api_error` at that node. The Checkpoint is kept so `/runs` can resume the run, nothing is applied, and the error type and `request_id` are logged. Failed requests have no `usage`, so they add no cost.
- **Chat policy:** the turn ends with a visible note, and the history is cut back to the last finished turn (as with Ctrl-C).
- **Bench:** an `api_error` run gets a ledger line, is counted in the report, and is left out of the Green means.
- **Config:** `[anthropic] max_retries` and `timeout`, passed to the client.

**Gates** (`tests/test_api_faults.py`):
1. A 429 or 529 followed by success completes, and the request count shows the retries.
2. Errors past `max_retries` end the run with `api_error`, a Checkpoint and no apply. Resuming from it with a healthy transport finishes the run.
3. A stream cut mid-turn in chat leaves the session consistent, and the next turn works.
4. `max_tokens` and `refusal` keep their F1 behavior (retry once, then raise with the category).
5. No API key appears in any error log line.

**Status (2026-10-11): built and gated** (`tests/test_api_faults.py`; `FaultyTransport` and `InstantRetries` in `tests/anthropic_fakes.py`). No live calls.

- **One failure type.** `harness/anthropic_api.py` has `ApiFailure` (the SDK's error type, HTTP status, `request_id`, and a message cut to 300 characters with any `sk-ant-…` redacted) and `make_client`, which both API paths now use. Whatever the SDK gives up on becomes an `ApiFailure`: its own `APIError`s, an `httpx2.TransportError` raised mid-stream (the SDK wraps transport errors only before the response starts), and a 200 whose body is not JSON (`json.loads` raises a plain `JSONDecodeError`, which the SDK neither wraps nor retries). It is not a `WorkflowModelError`, so no node mistakes it for invalid output, and the Plan loop's per-Critique catch does not swallow it.
- **Workflows.** The node wrapper (`timed`, in both loops) logs an `api_error` line (node, error, status, `request_id`, and the usage of any call that did succeed in that node) and re-raises. A fanned-out Critique has no wrapper, so it stops the run the same way without a line of its own. `drive_graph` catches it and returns the checkpointed values with `status: "api_error"` and an `api_error` record. The failed node wrote nothing, so the Checkpoint's `next` is that node and a resume runs it again. The CLI exits 4 (`API_ERROR_EXIT`) and says where it stopped, with the Workspace kept. The TUI marks the run `failed` (not `finished`), so `/build --resume` and `/plan --resume` accept it.
- **Chat.** A failed turn shows `[API error: <type> <status>, request <id>: …; the turn was dropped, send it again]` and is cut back with `settle`, as Ctrl-C does. Any other error in a turn is cut back the same way.
- **Bench.** An `api_error` run gets its ledger line as usual. `aggregate` leaves it out of every mean (and its repeat out of the total), and the label and comparison tables add a line counting such runs.
- **Config.** `[anthropic] max_retries` (default 2) and `timeout` (default 600 s, with the SDK's 5 s connect timeout kept).
- **Bugs the faults found.**
  1. A chat stream that ended before `message_stop` was accepted as a finished turn: `get_final_message()` returns the partial snapshot with `stop_reason: None`, which the model mapped to `end_turn`. That is now an `IncompleteStream` failure.
  2. A mid-stream connection reset and a non-JSON 200 raised raw `httpx2`/`json` errors that nothing expected.
  3. Strands' default retry strategy retried a 429 six more times, from 4 s up to 240 s apart, after the SDK had already given up. `build_api_agent` now passes `retry_strategy=None`, so the SDK's retries are the only ones.
- **Tests.** Retries are instant: `InstantRetries` overrides the SDK's `_calculate_retry_timeout` to return 0 and records what it would have waited, so the `retry-after: 7` test checks the real value.

---

## F9 — Trace: Route records and thinking summaries

Two things a learner wants to see after a run: why the graph went where it went, and what the model was thinking.

- **Route records** (ADR 0008): `gate()` wraps every conditional edge and writes `{node: "route", from, to, basis, field, value}`. For a model basis it adds `model`, `request_id`, `reason` and `thinking`. The summary gets `routes: {code: n, model: n, human: n}`.
- **Thinking display:** `thinking.display` is `omitted` by default on Haiku 5.5, Sonnet 5.5 and Opus 5.5, so the text comes back empty. `[anthropic] thinking_display = "summarized" | "omitted"` defaults to `"summarized"`. Display controls visibility only: thinking is billed the same either way.
  - `AnthropicWorkflowModel` keeps the summary instead of dropping it, and the run log stores it on each model node.
  - The chat agent shows it in the call Inspector.
  - Compaction still strips thinking blocks (F1b).
- **RunInspector:** shows Route records in line with the nodes, and each model node's thinking summary under its reply.

**Gates:**
1. With `"summarized"`, the request body carries `thinking: {type: "adaptive", display: "summarized"}`. With `"omitted"`, it carries no `display`.
2. A faked thinking summary reaches the run log and the RunInspector text.
3. Every conditional edge in a scripted run writes exactly one Route record, and the counts in the summary match the log.
4. No request carries sampling parameters (F1's gate still holds).

**Status (2026-10-11): built and gated** (`tests/workflow/test_trace.py`, 15 tests). No live calls.

- **Route records.** `records.Route` is what every route function now returns: `to`, the deciding state `field` and its `value`, and a `basis`. Each loop's `gate(source, route)` writes it with `RunLog.route` and counts it on `Recorder.routes` (checkpointed with the rest of the Recorder). Both loops are covered: the Execution loop's 15 conditional edges and the Plan loop's 4 (`research → draft`, the Critique fan-out, `synthesize → human_review`, `after_review`). Where the spec was silent:
  - A fixed edge that only the Spend cap can divert (`write_test → run_red`, `polish → verify_polish`, and so on) is a conditional edge in the graph, so it writes a record too, with `field: null`. A Spend cap diversion records `field: "cost_usd"` and the cost.
  - The basis is `human` on the edges an interrupt answer decides (after `clarify`, Stuck, Sign-off and `human_review`; the field is `answer`). In a no-hints run, code answers in the human's place, so those edges are `code`. `human_review → finalize` at the round limit is `code` (`field: "round"`). `finish` goes to END on a plain edge, so it writes none.
  - The fan-out records one route to `critique` with the angles as its value, not one per `Send`.
  - String values are cut to 200 characters. Records also carry `behavior` (Execution loop) or `round` (Plan loop).
  - Route records go to the log only. The RunLog listener, which drives the CLI progress lines and the TUI pane, still sees one record per node.
  - The summary's `routes` is counted from the run log when there is one. A Checkpoint is taken before the last node's route runs, so a resumed Recorder can be one route short.
- **F10's hook.** `Route.by_model(to, field, value, *, model, request_id, reason, thinking)`, returned from an `after_arbitrate` route function, writes a model-basis record and counts it under `model`. The arbitrate node gets `request_id` and `thinking` from its `ApiCall` (the model's `calls`) and puts them in the state with the Ruling, where the route function reads them.
- **Thinking.** `anthropic_api.thinking_param` builds the field: `{type: "adaptive", display: "summarized"}`, or nothing at all for `"omitted"` (the request is then as it was before F9). Both API paths send it. Each `ApiCall` has a `thinking` field, so the summary reaches a node's log line the way its usage does (`Recorder.add_usage`), with no change to the `WorkflowModel` protocol. The Ollama model has no `ApiCall`s and logs none. A retry's summaries are joined.
- **Inspectors.** The chat Inspector shows the summary above the request body. The session store's `calls` table gains a `thinking` column, added in place to an older store. The RunInspector lists Route records indented under the node they leave (`→ write_impl  code: red_verdict="red"`) and shows `--- thinking summary ---` after a node's reply. A model-basis route shows its model, request id, reason and thinking.
- **Compaction** still strips thinking blocks. The F1b test now strips a summarized one.
- **Existing tests changed.** The F1 "no `thinking` key" assertions now expect the summarized field, and four tests that compared the log one-to-one with the node trace now leave Route records out.

---

## F10 — The Arbiter: a stronger model settles uncovered Disputes

The F3 open question. When `check_dispute` doesn't uphold a Dispute (no Invariant is violated), and `[workflow] arbiter` names a model, an `arbitrate` node asks that model for a ruling.

- **Input:** the goal, the Plan with its Invariants, the Behavior, the disputed test's source, the Dispute's reason and the last pytest output. Not the implementation: the question is whether the test matches the goal.
- **Reply:** `Ruling {verdict: "test_wrong" | "test_right" | "unclear", reason, proposed_invariant: Invariant | null}`.
  - `test_wrong` takes the same path as an upheld Dispute: back to the `write_test` of the Behavior that owns the test, with the ruling as feedback.
  - `test_right` puts the ruling in the implementer's next prompt.
  - `unclear` goes to Stuck and shows the ruling.
- **Bound:** `max_arbitrations` per Behavior (default 1). Past it, the code route applies (the Dispute is not upheld).
- **Proposed Invariant:** recorded in the summary and shown at Stuck or Sign-off for the human to add to the Plan. It is never added automatically.
- **Models and cost:** add `claude-sonnet-5-5` and `claude-opus-5-5` to `PRICING`. `Price` needs an explicit cache-read price, because Opus 5.5's ($0.20) is not 0.1× its input price; check the prices at build time. The arbiter's calls count toward the Spend cap and are reported per role.
- **Refusals:** Opus and Sonnet 5.5 requests opt in to server-side fallbacks (`fallbacks: "default"` with its beta header). The model that actually served the request is recorded.
- **Stats:** `arbitrations`, `rulings_test_wrong` and `rulings_test_right`.

**Gates:**
1. A scripted `test_wrong` ruling sends the work to the right `write_test`, and the corrected test reaches Green.
2. `test_right` puts the ruling in the next `write_impl` prompt.
3. The bound holds: a second Dispute in the same Behavior takes the code route.
4. Every arbitration writes a model-basis Route record with `request_id` and reason.
5. The arbiter's prompt contains no implementation file.
6. With no `arbiter` set, the graph is unchanged (the F3 gates still pass).

**Live check:**
- `disputed-control` (no Invariants) with the arbiter on Sonnet 5.5, then Opus 5.5, against the F3 control. Does it rescue the six Stucks?
- `claude-plans`, where the tests are right, for the false-ruling rate.
- `exotic`, against `baseline-exotic`.

**Status (2026-10-11): built and gated** (`tests/workflow/test_arbiter.py`, 20 tests; the F3 gates in `test_execute.py` pass unchanged). No live calls.

- **Config.** `[workflow] arbiter` (an API model id), `arbiter_effort` and `max_arbitrations` (default 1). `config.arbiter_role(wf)` is the one reader of the first two and returns a `RoleModel(model, effort)`; F11 can point it at `[workflow.roles.arbiter]`. `backend.make_arbiter(config)` builds the Arbiter as an `AnthropicWorkflowModel`; an Ollama id or an unpriced model is refused before a run (the bench checks it with the other prices). `build_once` and the TUI build it from config; the summary's `execution` gains `/arbiter=<model>`.
- **Graph.** `build_execution_graph(..., arbiter=)` adds the `arbitrate` node only when an Arbiter is given; without one the nodes, edges and Route records are exactly F3's. `after_dispute` sends a Dispute no Invariant upheld to `arbitrate` while the state's `arbitrations` (reset when a new Behavior starts) is below the bound, else to `run_green`; both are code-basis Route records with `field: "arbitrations"`. `after_arbitrate` returns `Route.by_model(..., field="verdict")` with the model, `request_id`, reason and thinking summary, which the node copied from its `ApiCall`s (`UsageMeter.peek`).
- **Ruling.** `test_wrong` goes through `send_back`, the path an upheld Dispute takes (now shared with `check_dispute`): the owner's `write_test`, with a revisit for an earlier Behavior's test, and the attempt not counted. `test_right` replaces the "not upheld" note in the next `write_impl` prompt. `unclear` goes to Stuck, whose payload shows the Ruling under the Dispute; a hint returns to `write_impl`. Where the spec was silent:
  - The Arbiter gets the whole test file that holds the disputed test (found by path, then by function name), not the function alone, so imports and fixtures come with it. Its prompt has the Plan's Invariants whether or not `show_invariants` is on.
  - No valid Ruling (invalid JSON twice, or a refusal even after fallbacks) takes the code route (`field: "ruling"`): the Dispute stays not upheld. It still counts toward the bound.
  - `test_wrong` does not count in `disputes_upheld`, which stays "an Invariant upheld it". It clears the "not upheld" note.
- **Proposed Invariants** go into `Recorder.proposed_invariants` (checkpointed) and the summary's `proposed_invariants`, and into the Stuck and Sign-off payloads when there are any (the CLI and the TUI print them as "not applied"). Nothing adds them to the Plan.
- **Prices and cost.** `Price.cache_read` is an explicit cache-read price; None keeps 0.1× input, so Haiku is unchanged. Sonnet 5.5 is $2/$10, Opus 5.5 $4/$20, both with $0.20 cache reads, a 1M window and no long tier. Cache writes stay 1.25× input: neither the SDK nor the repo states a price for them, so that is the existing convention, unverified for these two models. A call is priced as the model that served it (`ApiCall.served_model`, from `response.model`) when that model is priced. The Arbiter's calls go through `add_usage`, so they count toward the Spend cap (a gate stops a run at the Spend cap right after an arbitration). The summary's `roles` has calls and cost per Role, by node: `write_test` is the tester, `arbitrate` the Arbiter, the rest the implementer. The CLI's final report adds a "by role" line when an Arbiter ran.
- **Fallbacks.** In SDK 1.13.0, `fallbacks` is a typed param of `client.beta.messages.create`, so `AnthropicWorkflowModel(fallbacks=True)` calls it with `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`, with no `extra_body`. It accepts the `BetaMessage` that comes back. `make_arbiter` sets `fallbacks` only for Sonnet and Opus 5.5; Haiku requests, the tester's and the implementer's, are unchanged. The `arbitrate` log line has `arbiter` (the model asked) and `served_by`, and the Route record names the model that served the request. In a batched bench, `cli.batched_arbiter` puts the Arbiter on the run's own `BatchLane`, which never sends fallbacks, and `batch_params` now strips `betas` as well as `fallbacks`.
- **Stats.** `arbitrations`, `rulings_test_wrong` and `rulings_test_right` per Behavior and in the summary's totals.

**Live checks: pending the user's approval.** Set `[workflow] arbiter` and run `harness-workflow bench <set> --model claude-haiku-5-5 --repeat 3 --label <label>` for each:
1. `disputed-control` with `arbiter = "claude-sonnet-5-5"`, then `"claude-opus-5-5"` (labels `f10-sonnet-control`, `f10-opus-control`), against `f3-control`.
2. `claude-plans` with the Sonnet Arbiter (`f10-sonnet-plans`), for the false-ruling rate.
3. `exotic` with the Sonnet Arbiter (`f10-sonnet-exotic`), against `baseline-exotic` (after F7's `make_katas.py`).

**Estimate.** An arbitration sends about 2–4k input tokens and gets 1–3k output tokens back, including thinking. That is about $0.02–0.04 on Sonnet and $0.04–0.08 on Opus. The F3 control had about two uncovered Disputes per repeat, and Haiku costs $0.018 per repeat, so check 1 is about $0.25 on Sonnet and $0.45 on Opus. Check 2 is about $0.10 of Haiku plus a few arbitrations, about $0.25. Check 3 is about $0.12 of Haiku plus arbitrations, about $0.40. That makes **about $1.35** in all; allow **$2.50**. The default $0.10 Spend cap per run fits one Opus arbitration, but a Behavior that disputes twice could pass it, so use `[workflow] max_cost_usd = 0.25` for the Opus run.

---

## F11 — Roles: separate context, model and view per role (adversarial TDD)

The test writer and the implementer become separate roles. Each has its own system prompt, model, effort and **view** (what context it gets).

- **Config:** `[workflow.roles.tester]`, `[workflow.roles.implementer]` and `[workflow.roles.arbiter]` (F10), each with `model`, `effort` and `view`. Unset fields fall back to `[workflow]`.
- **Views:**
  - `full`: today's context.
  - `interface`: the Plan, the current Behavior, earlier tests, and the implementation's *public interface* only. The interface is extracted with `ast`: signatures, docstrings and type hints, with no function bodies. This is the black-box tester.
  - The implementer always sees the tests (it must make them pass) but never the tester's rationale.
- **Prompts:** a tester system prompt about finding failure modes the goal implies, and an implementer prompt as today. Each role gets its own cache prefix.
- **Records:** `execution` names the roles' models and views. Cost and calls are recorded per role.
- **Prototype first** in `examples/graphs/roles_proto.py` with a scripted model, then port.

**Gates:**
1. With `view = "interface"`, no tester prompt contains a function body from the implementation (a canary body in a fake Workspace).
2. Each role's requests go to its own model (counted per model on the fake transport).
3. The summary carries cost and calls per role.
4. With no roles configured, prompts are byte-identical to today's (no regression to the cache prefix).

**Live check** (on `exotic`, 3 repeats each; the measures are hidden pass rate, self/hidden gap and mutants killed):
- (a) Haiku in both roles, `full` against `interface`.
- (b) Sonnet or Opus tester with a Haiku implementer.
- (c) Haiku tester with a Sonnet implementer.

The question: does a stronger or blind tester close the self/hidden gap more cheaply than a stronger implementer?

---

## F12 — Model frontier and Batches

- **Sweeps:** `bench <set> --models a,b,c --effort low,medium,high` runs the grid and labels each cell. `report --frontier <labels>` prints hidden pass rate against cost per kata. An HTML chart is optional.
- **Batches** (50% off): a Workflow's calls are sequential, so a batch per call would add hours to every node. Instead, `BatchedWorkflowModel` gathers the pending `complete()` calls from N concurrent bench runs, submits them as one batch once every active run is waiting (or after `flush_s`), and resolves each call by `custom_id`.
  - It fits a large `--repeat` run overnight. Each run's Spend cap uses batch prices.
  - Server-side fallbacks are rejected on Batches, so batched arbiter calls go without them.
  - Before building, check with one live request that `output_config.format` works in a batch.

**Gates:**
1. N scripted runs progress in lockstep through one fake batch endpoint, with results matched by `custom_id` in any order.
2. An `errored` or `expired` batch result follows F8's `api_error` path for that run only.
3. Cost is recorded at batch prices.
4. The sweep writes one label per cell, and the frontier report reads them.

**Live check:** `exotic` × {Haiku, Sonnet 5.5, Opus 5.5} × {low, medium}, 3 repeats, batched. Get the user's approval on the estimate first.

**Status (2026-10-11): built and gated** (`tests/workflow/test_batch.py`, 13 tests; the batch endpoints are faked by `FakeBatches` in `tests/anthropic_fakes.py`). No live calls.

- **Sweep.** `bench <set> --models a,b --effort low,medium` builds one Cell per model × effort (`cli.grid_cells`), labelled `<label>-<model>-<effort>` (the base label defaults to `<set>-<timestamp>`; a Cell with no `--effort` is `default`). Without `--models`/`--effort` the label is as before. Every model must be an Anthropic model with a price in `PRICING`, and every effort a known one: otherwise the bench exits 2 before any run, naming the model. `--max-cost` covers the whole grid. `run_bench` is now the one-Cell case of `run_grid`.
- **Frontier.** `report --frontier A B C` gives one row per label, cheapest first: model and effort (read from the summaries' provenance), runs, hidden pass rate (the mean over katas of each kata's mean rate) and mean cost per kata run. `*` marks a label no other label beats on both. Below that is a kata × label table of hidden rate · cost. A label with no graded runs shows `-` and is never on the frontier. There is no HTML chart.
- **Batches** (`workflow/batch.py`). A `BatchLane` stands in for the `AsyncAnthropic` client, so `BatchedWorkflowModel` is `AnthropicWorkflowModel` unchanged: it sends the same params as a single call (a gate compares the two bodies), keeps the retry-once on invalid JSON, and fails in the same way. `model.py` is not touched. The `Batcher` sends a batch once every open lane has a call waiting, or `--flush-s` (default 30 s) after the first waiting call. It polls every 15 s until `ended`, then resolves each call by `custom_id` (`run<k>-<n>`); the fake returns results in reverse order. A run's lane closes when its build ends, before grading, so the others stop waiting for it. A reply's `request_id` is `<batch id>/<custom_id>`.
  - **Failures.** An `errored`, `expired` or `canceled` result raises `ApiFailure` (`batch_errored:<type>`, `batch_expired`) for that call only, so that run ends with `api_error` at that node (F8) and the others go on. A batch request that fails past the SDK's retries fails every call in it. On Ctrl-C, the batches in flight are cancelled on the server.
  - **Cost.** Each call is priced by `api_call` and then halved (`batch.batch_priced`, `BATCH_DISCOUNT = 0.5`, not in `PRICING`), so the summary, the ledger and the run's Spend cap all use batch prices.
  - `batch_params` drops `fallbacks` (and the SDK's request options), so F10's fallbacks cannot reach a batch.
- **Concurrency.** With `--batch`, every run of the whole grid (all Cells, not one Cell at a time) goes at once, so a batch holds every waiting call. `--parallel N` limits that number; without `--batch` the default is 1, as before. Under concurrency, `--max-cost` still holds: each run starts with a Spend cap of the smaller of `max_cost_usd` and what is left after setting aside the caps of runs already going. A run that would get no cap waits for another run to end.
- **Merge note.** This touches `cli.py`'s bench (`run_bench` → `run_grid`/`bench_one`) and `bench.py`'s report, not `model.py`. F10 has since added `claude-sonnet-5-5` and `claude-opus-5-5` to `PRICING`, so a sweep naming them runs (it no longer exits 2 with "no pricing"). The estimate below used 20× and 40× Haiku's input and output prices, which the F10 prices confirm; Opus 5.5's cache reads are $0.20, not 40× Haiku's, so cache-heavy Opus runs cost a little less than estimated.

**Live checks: pending the user's approval.**
1. **`output_config.format` in a batch** (unverified). Send one Haiku request through `BatchedWorkflowModel` with any schema: one batch, about $0.0001. If it comes back `errored`, the batched path needs a different way to get JSON before the sweep.
2. **The sweep:** `harness-workflow bench exotic --models claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5 --effort low,medium --repeat 3 --batch --label frontier` (after F10's prices and F7's `make_katas.py`), then `report --frontier frontier-claude-haiku-5-5-low ...`. That is 36 runs.
   - **Estimate.** `baseline-f1` cost $0.0334 per repeat of 5 katas, or $0.0067 per kata run on Haiku. F3's disputed Plans cost $0.0050–0.0059 per run. Six exotic katas therefore cost about $0.040 per repeat on Haiku, or $0.12 per Cell at standard prices. Sonnet 5.5's input and output prices are 20× Haiku's and Opus 5.5's are 40×, so a Cell costs about $2.40 on Sonnet and $4.80 on Opus.
   - For two efforts that is about $14.6 at standard prices, or **about $7.3 batched**. Allow up to about **$15**: the exotic katas are new and harder (F7 has no cost yet), batch cache hits are best effort, and effort changes thinking tokens.
   - **Settings needed.** An Opus run would cost about $0.13 batched, which is over the default $0.10 Spend cap. Set `[workflow] max_cost_usd = 0.50`, and pass `--max-cost 20` so that all 36 runs fit under the bench's reservations at once.
   - **Time.** Each run makes about 20–30 sequential calls, and each call costs one batch round. Plan for it overnight.

---

## F13 — Same task, four frameworks (an example, not a feature)

In `examples/frameworks/`, one exotic kata runs through a test writer and an implementer in four frameworks:

1. braid's LangGraph Workflow (F11 roles).
2. A Strands `Swarm` with handoffs.
3. The Claude Agent SDK with subagents.
4. CrewAI, run with `uv run --with crewai` so it never becomes a project dependency.

Each is graded by F7's hidden tests and reports its own token usage.

- **Write-up:** a table of hidden pass rate, cost, calls and lines of glue code, plus what each framework made easy or hard: explicit routes against emergent handoffs, and where the trace lives.
- **Isolation:** frameworks 2–4 run their own tool loops, so model-written code runs outside braid's Sandbox. They run in a temporary copy of the kata with the API key kept out of tool environments where the framework allows it. The README says so. This is outside ADR 0005, which covers the harness. Nothing from here goes into `src/`.

**Gate:** only the grading adapter (each runner's output directory → hidden grade) gets a test.

---

## Open questions

- F1: does Haiku's adaptive thinking need `effort: low` for the katas, or is the default `medium` worth its cost? Measure both on one kata.
- F1: should the Plan loop default to Haiku once F1 lands, since local Plans are weak? That decision belongs to the user after the live check.
- F3: when no invariant covers a disputed test, should the work go back to `write_test` once per Behavior anyway? **Decided 2026-10-11:** routing on a model's claim is allowed when it is traced and bounded (ADR 0008). F10 tries an Arbiter first; the Dispute alone can be a bench variant.
- F7+: should F7–F9 (no dependency on F4–F6) go ahead of F4–F6? **Decided 2026-10-11:** F4–F6 are skipped for now. The work order is F7–F13.
- Later: `write_change` (test and implementation in one reply). Haiku made 0 out-of-scope edits in 103 write calls, against e4b's 27 in 7 runs, so it only matters if e4b stays a build target. F4's mixed-edit handling covers e4b more cheaply.
