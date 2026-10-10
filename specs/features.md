# Features after M15

M11–M15 are built (see `specs/initial.md` status notes). These are the next features, planned 2026-10-10 from the live results: e4b gets 28/30 Green with Claude's Plans, but only 11/24 with its own; mixed edits cost attempts; small-model planning does not converge. Terms are from `CONTEXT.md`. Each feature lists its gates; every gate gets a pytest test with fakes, and live numbers are measured, not gated.

**Order:** F1 → F2 → F3 → F4 → F5. F1 comes first because a Haiku row turns every later change into a three-way comparison (e4b, Haiku, Claude Plans) at a few cents per run.

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

**Later, not in F1:** Haiku as the Plan-loop model (Claude-quality Plans without `claude -p`). The chat agent on Strands' Anthropic provider (allowed by ADR 0006; build it when it is needed). Anthropic Batches for a cheap overnight baseline.

---

## F2 — `write_change`: test and implementation in one reply

The live runs showed that models want to write the test and the code together. Splitting them across `write_test` and `write_impl` caused every mixed-edit refusal. Keep the guarantee, drop the ritual.

- Replace `write_test` → `run_red` → `check_satisfied` → `write_impl` → `run_green` with `write_change` → `check_red` → `run_green`.
- `write_change` returns `Edits` with the new test file(s) and the implementation change together.
- `check_red` runs only the new test(s) against the Workspace **as it was before this change**: it restores the pre-change contents of every non-test file, runs the tests, then restores again in a `finally` (the `check_satisfied` machinery). The new test must fail there for an expected reason, which is the existing Red rule (exit 1, or an import error naming a non-test module). If it passes there, it is vacuous or Satisfied, and `check_satisfied`'s HEAD comparison decides which.
- `run_green` runs the whole suite as now. On failure, the next attempt may edit the implementation and the new test, but never earlier Behaviors' tests (from the semver-compare live run: a wrong earlier test should go to Stuck with a hint, not be silently rewritten).
- **Build it as a prototype first** in `examples/graphs/` and run two or three katas on e4b, plus Haiku once F1 lands, against the current graph. Port it into `workflow/execute.py` only if Green is at least as good with fewer model calls. Keep the split graph behind `[workflow] execution = "split" | "change"` until the numbers decide.

**Gates** (`tests/workflow/test_execute.py`, extended):
1. A change whose test fails before and passes after reaches `next_behavior` in one model call.
2. A test that passes before the change goes to the vacuous-or-Satisfied decision. The two branches are the same as gates 2 and 2b.
3. An edit to an earlier Behavior's test is refused and counts as an attempt.
4. The pre-change restore leaves the Workspace byte-identical, even when the test run raises.

---

## F3 — Mixed edits and Plan checks

**Mixed edits** (still needed for Polish and the split graph). When a reply mixes allowed and refused files, write the allowed files, **still run the check**, and put the refusal in the next prompt. Gate: a `write_impl` reply with the implementation plus a test file runs `run_green` on the implementation and does not use up a second attempt for the refusal alone.

**Plan checks after `draft`.** These are deterministic checks, and a failure goes back to `draft` with the reason (at most once, as with Assumptions):
- Every path in `files` exists in the project, or is marked new (a new `new_files` list, or a `(new)` suffix).
- `demo` is a valid command: for `python -c`, the code argument compiles.
- **A Probe must be able to fail.** Run it once as written (it should pass), then once against a Workspace whose module is emptied back to its docstring. A Probe that passes both times proves nothing; flag it, and after the one retry drop the Assumption and make it an open question.

**Review convergence.** `synthesize` sees the questions already asked and answered, and returns only new ones. A round with no new questions offers `approve` as the default. Gate: answered questions do not come back in the next round.

**Gates** (`tests/workflow/test_plan_loop.py`, extended): a Plan with a missing file, an invalid demo or a probe that cannot fail gets exactly one draft retry naming the problem; answered questions are not asked again.

---

## F4 — Time travel: fork a run from a checkpoint

LangGraph keeps every checkpoint (M15), so a stuck run can be replayed from just before its bad node with a different prompt, hint or model.

- `/runs <id>` lists the run's checkpoints (node, Behavior, attempt).
- `/runs <id> --fork <checkpoint>` creates a new run id from that checkpoint's state, in a fresh Workspace: a new worktree with the checkpointed file contents written in. It optionally takes `--model` and `--hint`. The original run is never changed.
- The CLI gets the same command: `harness-workflow fork <run-id> <checkpoint> [--model] [--hint]`.

**Gates:** a fork replays from its checkpoint without repeating earlier model calls (count `ScriptedModel` calls); the original run's checkpoint and Workspace are unchanged; a fork with `--model` uses that model for its next node only from that point.

**Try it:** fork the semver-compare Behavior 3 run from M13 on Haiku and see whether it gets past the contradictory test.

---

## F5 — Graphs defined in YAML, as skills

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
- F2: whether the split graph stays as an option once `write_change` wins, or is deleted.
