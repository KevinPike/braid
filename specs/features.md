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

**Gates:** each load-time check rejects a bad file with a message naming the node; a valid file compiles and runs to its first interrupt with a scripted model; a condition on a model field is rejected at load time.

---

## Open questions

- F1: does Haiku's adaptive thinking need `effort: low` for the katas, or is the default `medium` worth its cost? Measure both on one kata.
- F1: should the Plan loop default to Haiku once F1 lands, since local Plans are weak? That decision belongs to the user after the live check.
- F3: when no invariant covers a disputed test, should the work go back to `write_test` once per Behavior anyway? That would route on a model's claim, so it needs an ADR 0004 amendment. Another option is an arbiter on a stronger model (a crew). Decide after the F3 replays.
- Later: `write_change` (test and implementation in one reply). Haiku made 0 out-of-scope edits in 103 write calls, against e4b's 27 in 7 runs, so it only matters if e4b stays a build target. F4's mixed-edit handling covers e4b more cheaply.
