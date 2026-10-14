# Local agent harness

Python 3.12 TUI (Textual) and LangGraph Workflows on this machine, with models on the Anthropic API (default `claude-haiku-5-5`). The Ollama path and its Guard are frozen (ADR 0007): kept green, with no new work.

## Read first

- **Plan and milestone status**: `specs/initial.md`. Milestones are zero-indexed; M0 is done. Work the next unticked milestone and meet its validation gate before starting another.
- **After M15**: `specs/features.md` (F1, F1b, F2, F3, then F7–F13, in order; F4–F6 are skipped for now).
- **Vocabulary**: `CONTEXT.md`. Use its terms (Workflow, Behavior, Plan, Spend cap, Invariant) in code, tests and commits.
- **Why a design is shaped this way**: `docs/adr/`. Surface a conflict with an ADR instead of overriding it.

## Commands

`uv run harness` runs the app; `uv run pytest -q`, `uv run mypy src tests` and `uv run ruff check .` must all pass before committing (`mypy --strict`, full type hints; ruff rules are in `pyproject.toml`).

## Conventions

- Every validation gate in the plan gets a `pytest` test. Unit tests use fakes and need neither Ollama nor the network; fake the Anthropic API with `httpx2.MockTransport` (respx cannot see SDK 1.x traffic).
- Live checks run on `claude-haiku-5-5` with `uv run --env-file .env`, and only after the user approves the spend.
- `guard/` never imports `strands` or `anthropic` (ADR 0001, 0006).
- Every Ollama request goes through the guard client, which sets `num_ctx`; a second path would cause reloads. Workflows use `langgraph` with no `langchain-*` model class (ADR 0004).
- Code a model wrote runs only through the `Sandbox` interface; git and `uv sync` belong to the harness and run outside it (ADR 0005).
- Strands agents are built with `callback_handler=None`; the TUI renders the stream, stdout stays clean.
- The app binds `ctrl+c` with `priority=True` to cancel generation instead of quitting.
- **Frozen Ollama path**: before touching `guard/`, Calibration, Preflight or an Ollama-backed run, read `docs/ollama-machine-facts.md`.

## README voice

The README is a pitch, not a manual. When you edit it, keep this shape and tone.

- **Order**: one-line promise, the idea in a sentence, a loop diagram, the sandbox, cost control, a three-line quickstart, then "Also in the box" and links. Reference material (keys, slash commands, shell usage, config, telemetry, tests) goes in `docs/usage.md`, not the README.
- **Voice**: dry and understated, like Tailscale. Lead with what the reader gets. Back each claim with a number or a mechanism ("a few cents", "no network", "stops at the dollar limit you set"). No hype words: seamless, robust, powerful, effortless, blazing.
- **Sentences**: short, active, one idea each. No semicolons. Verbs over nouns ("analyze", not "perform an analysis"). No phrasal verbs. This is the structural part of ASD-STE100 only, not the dictionary.
- **Terms**: use the `CONTEXT.md` words and use each one the same way every time.
- **Truth**: do not claim what the code does not do. A new feature goes in "Also in the box" with one line, and the detail goes in `docs/usage.md`. Do not invent run output in the quickstart.
