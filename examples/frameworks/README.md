# Same task, four frameworks (F13)

An example, not a feature. One exotic kata (`bench/exotic/<kata>`) is built by a **tester** and an **implementer** in
four frameworks, and each result is graded the same way, by F7's Hidden tests and mutants:

1. **braid**: the Execution loop, a LangGraph Workflow, with F11's Roles (`run_braid.py`).
2. **Strands `Swarm`**: two agents that hand off to each other (`run_strands_swarm.py`).
3. **Claude Agent SDK**: a coordinator with a tester subagent and an implementer subagent (`run_agent_sdk.py`).
4. **CrewAI**: two agents with one task each, run in sequence (`run_crewai.py`).

The question is what each framework makes easy or hard for this one job: **explicit routes** (braid's graph decides
every step in code; CrewAI's sequential process is a fixed line) against **emergent handoffs** (in the Swarm and the
Agent SDK the model decides when to hand off and whether to go back), and **where the trace lives**.

## Isolation: read this before running 2–4

> **Runners 2, 3 and 4 run model-written code outside braid's Sandbox.** Their frameworks run their own tool loops,
> so pytest runs on the model's module and tests as an ordinary process on this machine. This is outside ADR 0005,
> which covers the harness, not these examples. **Nothing here goes into `src/`.**

What holds instead (`common.py`):

- Each run works in a **fresh temporary copy of the kata repo** (`GOAL.md`, a stub module, `pyproject.toml`,
  `tests/`), made by `examples/graphs/make_katas.py` from `GOAL.md` and `summary.txt` only. No Hidden test, reference
  or mutant is in it, and the tools cannot reach `bench/exotic`.
- The tools refuse any path outside that copy or in a dot-directory. Each Role writes only its own files: the
  tester writes test files, and the implementer writes everything else.
- **The API key is kept out of tool environments** where the framework allows it:

| runner | where the key is | tool environment |
|---|---|---|
| braid | the harness process | the Sandbox builds the environment from scratch, with no key, network off, writes only in the Workspace (ADR 0005) |
| Strands Swarm | this process (the 1.x client) | the tools are this example's; pytest runs with `env_without_keys()` (no `ANTHROPIC_*`, `*_API_KEY`, `*TOKEN*`, cloud credentials) |
| Agent SDK | the bundled Claude Code CLI (it makes the calls) | no Bash tool at all; pytest is an in-process MCP tool with `env_without_keys()`; `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` strips credentials from anything else the CLI starts |
| CrewAI | this process (the provider reads `ANTHROPIC_API_KEY`) | the tools are this example's; pytest runs with `env_without_keys()` |

The model-written code can still read files your user can read and reach the network. Run 2–4 only on a machine where
that is acceptable. The grading itself (`grade.py`) runs in the Sandbox, as `harness-workflow bench` grades.

## Running

Every runner takes `--kata <name> --out <dir>` and `--model` (default `claude-haiku-5-5`). It writes the final module
and tests into `--out`, its own trace into `<out>/.trace/`, and prints **one JSON line** on stdout: calls, tokens
(uncached input, output, cache reads and writes), `cost_usd` with its source, calls and cost per Role, status and the
trace's location. Live runs cost money: get the spend approved first (see the estimate in `specs/features.md`, F13).

```
uv run --env-file .env python examples/frameworks/run_braid.py         --kata ferry-fare --out runs/braid
uv run --env-file .env python examples/frameworks/run_strands_swarm.py --kata ferry-fare --out runs/strands
uv run --env-file .env --with claude-agent-sdk python examples/frameworks/run_agent_sdk.py --kata ferry-fare --out runs/agent-sdk
uv run --env-file .env --with crewai python examples/frameworks/run_crewai.py --kata ferry-fare --out runs/crewai

uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/strands
```

`claude-agent-sdk` and `crewai` are added for one run by `uv run --with`, so they never become project dependencies.

`grade.py` prints `hidden_passed`/`hidden_total`, `mutants_killed`/`mutants_total` (the runner's own tests against
F7's five mutants; 0 when it wrote none) and `own_tests`. It reuses `harness.workflow.bench.grade` on a staging copy of
the output directory, so it works on any directory, not just a braid Workspace. A missing module scores 0 of the whole
hidden suite. Its test is `tests/test_frameworks_grade.py`, the only gate F13 has.

### Per runner

- **braid** (`run_braid.py`). `cli.build_once` on `bench/exotic/<kata>/plan.json`, `--no-hints`, `no` at finish, as a
  bench run does. If `braid.toml` does not separate the Roles, the tester gets the `interface` View
  (`--tester-view full` to change it). The kata repo is made with `uv sync`, since a Workspace needs `uv.lock`.
- **Strands Swarm** (`run_strands_swarm.py`). Strands' own Anthropic provider pins `anthropic<1.0`, so each agent runs
  on braid's `AnthropicChatModel` (`src/harness/agent.py`), which reuses Strands' formatting, sends on the 1.x client,
  caches the history prefix and never sends sampling parameters. `max_handoffs=6`, `max_iterations=8`.
- **Claude Agent SDK** (`run_agent_sdk.py`, checked against `claude-agent-sdk` 0.2.165). Tools: Read, Write, Edit,
  Glob, the Agent tool and an MCP `pytest` tool. A `PreToolUse` hook confines paths and enforces each Role's writes
  (it knows the calling subagent from `agent_type`). `setting_sources=[]`, `strict_mcp_config`, `permission_mode="dontAsk"`,
  `max_budget_usd=0.50`. The CLI builds its own requests; nothing in the options sets a sampling parameter, and the CLI
  supports the 5.5 models. Without `ANTHROPIC_API_KEY` the CLI falls back to the machine's Claude login, and its cost
  figure is then notional. This runner was checked offline only up to its options and hook: the CLI calls the API as
  soon as it starts, so there was no offline end-to-end run.
- **CrewAI** (`run_crewai.py`, checked against `crewai` 1.15.27). It reaches Claude 5.5 through CrewAI's native
  Anthropic provider (`LLM(model="anthropic/claude-haiku-5-5")`, not LiteLLM). That provider's extra pins
  `anthropic~=0.73`, but under `uv run --with crewai` the project's `anthropic` 1.x is installed, and an offline run on
  a mock transport worked: the body is `model`, `messages`, `max_tokens`, `stop_sequences`, `stream` and `tools`. Its
  `temperature` and `top_p` default to None and are sent only when set; the runner refuses to start if either is set.
  CrewAI adds no prompt caching. Telemetry and tracing are turned off.

The Strands and CrewAI runners were each run offline end to end on a scripted fake API (no network), and their output
was graded under Seatbelt. Those were smoke checks of the plumbing, not results. `run_braid.py` calls the same
`build_once` the bench and its tests drive.

## Write-up

| | braid (LangGraph) | Strands Swarm | Claude Agent SDK | CrewAI |
|---|---|---|---|---|
| hidden pass rate ² | 45/46 (fails the child-on-a-bike case) | 46/46 | 46/46 | 46/46 |
| mutants killed (own tests) | 4/5 (6 test files) | 5/5 (1 file) | 5/5 (1 file) | 5/5 (1 file) |
| cost | $0.0076 | $0.0058 | $0.0107 (the CLI's total) | $0.0065 (approximate) |
| calls | 13 (6 tester, 7 implementer), 63 s | 7, one handoff, 34 s | 11 (3 coordinator, 5 tester, 3 implementer), 61 s | 7, 47 s |
| lines of glue code ¹ | 69 | 86 | 141 | 107 |
| routing | explicit: the graph routes on test results and Invariants, in code | emergent: `handoff_to_agent`, bounded by `max_handoffs` | emergent: the coordinator model calls the Agent tool | explicit and linear: `Process.sequential`, with no way back to the tester |
| easy | per-Role cost, a Route record for every step, the Sandbox, a Plan to bench against | the handoff tool and shared context, with little glue | built-in tools and a cost figure | a sequential pipeline with almost no wiring |
| hard | a fixed Plan can carry a misreading into every run (the bike rule) | reaching Claude 5.5 needed braid's own model class | restricting it: a hook, an allow-list, an MCP pytest tool; per-message usage under-counts | no prompt caching, and no way back to the tester |
| where the trace lives | `~/.braid/runs/<run>.jsonl` (every node, route, prompt and call) and `<out>/.trace/summary.json` | `<out>/.trace/swarm.json` (handoff order, each agent's messages) | `<out>/.trace/messages.jsonl`, and the CLI's transcript under `~/.claude/projects/` (session id in the JSON line) | `<out>/.trace/crew.json` (this example's tool log by Role, each task's output) |

¹ Non-blank, non-comment lines outside docstrings in each runner. Runners 2–4 also share about 90 lines of
`common.py` (the tools, the Role prompts and the key-free environment), and all four share its kata copy and JSON
line. braid's own Workflow is not counted: it is the harness.

² One run each on `ferry-fare` with `claude-haiku-5-5`, 2026-10-12, $0.0305 for all four. One run is an anecdote, not
a measurement. braid's miss is the one `baseline-exotic` makes in all 3 of its runs: its fixed Plan's only error example is
a van, and the implementer extended "needs an adult" to bikes. The other runners read the whole goal at once.

What is known before a live run:

- **braid** needs no glue for the roles, the Sandbox, the per-Role cost or the trace: they are the harness. It is
  also the only runner whose Plan (the Behaviors) is written in advance, so it does more structured work per kata.
- **Strands** needed a model class of braid's own to reach Claude 5.5 on the 1.x SDK. Its handoff tool and shared
  context come free, and the model decides the route.
- **The Agent SDK** has the richest built-in tools and usage reporting (`total_cost_usd`), but the most glue to
  restrict them: a hook, a tool allow-list, settings isolation and an MCP tool for pytest.
- **CrewAI** has the least control over the request: the runner checks after the fact that no sampling parameter is
  set, and caching is unavailable.
