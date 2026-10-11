# braid

A harness on your Mac for exploring Workflows as graphs: a Textual terminal UI over a Strands chat agent, and LangGraph Workflows that plan a feature and build it test-first in a sandboxed git worktree. Models run on the Anthropic API by default (`claude-haiku-5-5`, a few cents per run). The original local path, Ollama with a runtime guard for context and GPU memory, is frozen but kept ([ADR 0007](docs/adr/0007-anthropic-is-the-default-backend-ollama-is-frozen.md)).

Plans and status live in [`specs/features.md`](specs/features.md) (and [`specs/initial.md`](specs/initial.md) for M0–M15). Terms such as Workflow, Behavior, Plan, Invariant, Dispute and Spend cap are defined in [`CONTEXT.md`](CONTEXT.md), and design decisions are recorded in [`docs/adr/`](docs/adr/).

## Features

- **Workflows as graphs.** The Plan loop turns a goal into an approved Plan (research, Critiques, your review). The Execution loop makes each Behavior Red then Green, asks for Sign-off, then runs Polish. Models write and code decides: every route is a pytest run, a Probe, a path rule, an Invariant check or a human, never a model's claim ([ADR 0004](docs/adr/0004-workflows-are-langgraph-over-the-guard-client.md)). Runs are checkpointed, so a cancelled run resumes where it stopped.
- **Invariants and Disputes.** A Plan can state rules (`Invariant`s) as Python expressions over one call of the function under test. Code checks every expected value a test asserts against them, after Red and whenever the implementer disputes a test. A test that breaks a rule goes back to `write_test`, with the rule named.
- **Benchmarks.** `harness-workflow bench` runs a named set of katas N times into a ledger. Every summary records the harness commit, Plan hash and settings, and `harness-workflow report --compare A B` tells a real change from noise.
- **Sandbox.** Model-written code runs only under macOS Seatbelt, in a throwaway git worktree, with no network and a scrubbed environment (the API key never reaches it). Its diff reaches your project only through an Approval ([ADR 0005](docs/adr/0005-sandboxed-workspace-moves-approval-to-the-diff.md)).
- **Chat TUI.** Streaming Markdown, a prompt queue, approval prompts for writes and shell commands, sessions you can resume and fork, and a transcript inspector. The status bar shows the model, tokens against its real context window, cache reads and the session's running cost. Compaction summarizes old turns and strips thinking blocks from the history it rewrites, because Claude rejects edited history that replays them.
- **Cost control.** A Spend cap per Workflow run (`max_cost_usd`) and per bench (`--max-cost`). Prompt caching on the system prompt and history.

## Prerequisites

- macOS and [`uv`](https://docs.astral.sh/uv/) (`brew install uv`); it fetches Python 3.12 itself
- Anthropic credentials: `ANTHROPIC_API_KEY`, or `ant auth login`. The harness never reads a key from its config. It does not load `.env` itself either; `uv run --env-file .env ...` does.
- For the frozen local path only: [Ollama](https://ollama.com) 0.35+ and the models in `harness.toml`, with `[profile] backend = "ollama"`

## Run it

```sh
uv sync
uv run --env-file .env harness                 # the chat TUI; reads ./harness.toml
```

In the TUI, `/plan <goal>` runs the Plan loop and `/build [plan-id]` builds a Plan in the current project; `/runs` lists runs and `/build --resume <run-id>` continues one.

| Key | Action |
|---|---|
| Enter | send the message (on an empty input, resume a paused queue) |
| Up / Down | recall earlier prompts |
| Ctrl-C | cancel the current generation or pause the Workflow run (the app keeps running) |
| Esc | clear the prompt queue |
| Ctrl-T | transcript inspector |
| Ctrl-Q | quit |

Slash commands: `/model [name]`, `/compact`, `/trims`, `/tools`, `/sessions`, `/resume <id>`, `/fork [id]`, `/new`, `/inspect [n]`, `/plan`, `/build`, `/runs`, `/exit`.

From the shell:

```sh
uv run --env-file .env harness-workflow plan "<goal>" --project ~/code/thing
uv run --env-file .env harness-workflow build --plan plan.json --project ~/code/thing [--no-hints]

uv run --env-file .env harness-workflow bench claude-plans --repeat 3 --label my-change   # sets: bench/katas.toml
uv run harness-workflow report                       # labels in the ledger
uv run harness-workflow report --compare baseline-f1 my-change
uv run harness-workflow report --summaries ~/projects/braid-practice   # latest run per model x kata
```

The bench katas are small git repos made by `examples/graphs/make_katas.py` (in `~/projects/braid-practice` by default).

## Configure

Everything is in `harness.toml`: `[profile]` for the chat agent (backend, model, effort, system prompt), `[workflow]` for the Workflows (backend, model, Spend cap, attempts, test command, the F3 switches `show_invariants` and `check_tests_at_red`), `[context]` for compaction, `[tools]` for the chat's tools, and `[paths]` for the data directory (`~/.braid`: sessions, trim log, run logs, checkpoints and the bench ledger). Missing keys fall back to the defaults in `src/harness/config.py`.

## Telemetry and evals

**OpenTelemetry** is off by default. Start a local OTLP/HTTP collector and set `[telemetry] enabled = true` in `harness.toml`:

```sh
docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one              # traces at http://localhost:16686
```

**Evals** (`harness-eval`) compare Ollama models on hand-labelled tool-selection prompts; they belong to the frozen local path.

## Tests

```sh
uv run pytest -q          # unit tests use fakes: no network, no Ollama, no API spend
uv run mypy src tests     # strict type check
uv run ruff check .
```

## License

[Apache 2.0](LICENSE)
