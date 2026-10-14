# Usage reference

The README has the pitch and the quickstart. This page has the rest.

## Prerequisites

- macOS and [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). It fetches Python 3.12 itself.
- Anthropic credentials: `ANTHROPIC_API_KEY`, or `ant auth login`. For OpenRouter models, also set `OPENROUTER_API_KEY`. braid never reads a key from its config. It does not load `.env` itself either. `uv run --env-file .env ...` does.
- For the frozen local path only: [Ollama](https://ollama.com) 0.35+ and the models in `braid.toml`, with `[profile] backend = "ollama"`.

## The chat TUI

```sh
uv sync
uv run --env-file .env harness     # reads ./braid.toml
```

`/plan <goal>` runs the Plan loop. `/build [plan-id]` builds a Plan in the current project. `/runs` lists runs. `/build --resume <run-id>` continues one.

| Key | Action |
|---|---|
| Enter | Send the message. On an empty input, resume a paused queue. |
| Up / Down | Recall earlier prompts. |
| Ctrl-C | Cancel the current generation or pause the Workflow run. The app keeps running. |
| Esc | Clear the prompt queue. |
| Ctrl-T | Open the transcript inspector. |
| Ctrl-Q | Quit. |

Slash commands: `/model [name]`, `/compact`, `/trims`, `/tools`, `/decisions`, `/choices`, `/skills`, `/mcp`, `/sessions`, `/resume <id>`, `/fork [id]`, `/new`, `/inspect [n]`, `/plan`, `/build`, `/runs`, `/exit`.

The status bar shows the model, tokens against its real context window, cache reads and the running cost of the session. Compaction summarizes old turns. It strips thinking blocks from the history it rewrites, because Claude rejects edited history that replays them.

## The shell

```sh
uv run --env-file .env harness-workflow plan "<goal>" --project ~/code/thing
uv run --env-file .env harness-workflow build --plan plan.json --project ~/code/thing [--no-hints]

uv run --env-file .env harness-workflow bench claude-plans --repeat 3 --label my-change   # sets: bench/katas.toml
uv run --env-file .env harness-workflow models                 # the OpenRouter catalog
uv run --env-file .env harness-workflow build --plan plan.json --project ~/code/thing --backend openrouter --model deepseek/deepseek-v4-flash:nitro
uv run braid-mcp                                   # braid as an MCP server (stdio)
uv run harness-workflow report                       # labels in the ledger
uv run harness-workflow report --compare baseline-f1 my-change
uv run harness-workflow report --summaries ~/projects/braid-practice   # latest run per model x kata
```

The bench katas are small git repos made by `examples/graphs/make_katas.py`. They live in `~/projects/braid-practice` by default.

## Features in detail

- **Invariants and Disputes.** A Plan can state rules (`Invariant`s) as Python expressions over one call of the function under test. Code checks every expected value that a test asserts against them, after Red and whenever the implementer disputes a test. A test that breaks a rule goes back to `write_test`, with the rule named.
- **Benchmarks.** `harness-workflow bench` runs a named set of katas N times into a ledger. Every summary records the harness commit, Plan hash and settings. `harness-workflow report --compare A B` tells a real change from noise.
- **Sandbox.** Model-written code runs only under macOS Seatbelt, in a throwaway git worktree, with no network and a scrubbed environment. The API key never reaches it. Its diff reaches your project only through an Approval ([ADR 0005](adr/0005-sandboxed-workspace-moves-approval-to-the-diff.md)).
- **OpenRouter.** `--backend openrouter --model vendor/model` runs a Workflow, or a bench, on another vendor's model. A `vendor/model` id picks the backend by itself. The `:nitro` suffix is the fast route. `harness-workflow models` lists the catalog with price, context and structured-output support. Models without structured output are refused. The planner and the coder can differ (`plans-for`, then `bench --plans-dir`). See [ADR 0009](adr/0009-openrouter-is-a-third-backend-behind-the-workflow-model-seam.md) and [`specs/openrouter.md`](../specs/openrouter.md).
- **Choice.** The chat agent can ask you to pick from options (`ask_user_question`, [ADR 0010](adr/0010-a-choice-is-an-answer-not-an-approval.md)). `/choices` lists them. `uv run harness --demo choice` shows it offline.
- **Skills.** Agent Skills folders (`SKILL.md`) in `.braid/skills` or `~/.braid/skills`. Names and descriptions go in the system prompt. The body loads on demand or with `/skill-name`. Scripts run in the Sandbox with an Approval ([ADR 0011](adr/0011-skills-are-untrusted-instructions.md)). `/skills` lists them.
- **MCP client.** Servers in `[mcp.servers.*]` give the chat agent more tools. Every call needs an Approval unless allow-listed. Stdio servers run in the Sandbox ([ADR 0012](adr/0012-mcp-servers-are-sandboxed-gated-tool-providers.md)). `/mcp` lists and manages them.
- **MCP server.** `braid-mcp` exposes braid to a client such as Claude Code. Reads are free. Plan, build and bench runs start only after a human says yes, under per-run and daily spend ceilings. It can never give a Sign-off or apply a diff ([ADR 0013](adr/0013-braid-as-an-mcp-server.md)).
- **Cost control.** A Spend cap per Workflow run (`max_cost_usd`) and per bench (`--max-cost`). Prompt caching on the system prompt and the history.

## Configure

Everything is in `braid.toml`:

- `[profile]`: the chat agent (backend, model, effort, system prompt).
- `[workflow]`: the Workflows (backend, model, Spend cap, attempts, test command, and the F3 switches `show_invariants` and `check_tests_at_red`).
- `[openrouter]`, `[context]` (compaction), `[tools]` (the chat's tools).
- `[ask]`, `[skills]`, `[mcp]` and `[mcp_server]`: Choice, Skills and MCP.
- `[paths]`: the data directory (`~/.braid`) with sessions, trim log, run logs, checkpoints and the bench ledger.

Missing keys fall back to the defaults in `src/harness/config.py`.

## Telemetry and evals

**OpenTelemetry** is off by default. Start a local OTLP/HTTP collector and set `[telemetry] enabled = true` in `braid.toml`:

```sh
docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one              # traces at http://localhost:16686
```

**Evals** (`harness-eval`) compare Ollama models on hand-labelled tool-selection prompts. They belong to the frozen local path.

## Tests

```sh
uv run pytest -q          # unit tests use fakes: no network, no Ollama, no API spend
uv run mypy src tests     # strict type check
uv run ruff check .
```
