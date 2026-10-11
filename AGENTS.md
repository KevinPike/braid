# Local agent harness

Python 3.12 TUI (Textual) over a Strands agent on local Ollama, with a runtime guard for context and memory health.

## Read first

- **Plan and milestone status**: `specs/initial.md`. Milestones are zero-indexed; M0 is done. Work the next unticked milestone and meet its validation gate before starting another.
- **After M15**: `specs/features.md` (F1–F6, in order).
- **Vocabulary**: `CONTEXT.md`. Use its terms (Guard, Spill, Truncation, Pinned num_ctx) in code, tests and commits.
- **Why a design is shaped this way**: `docs/adr/`. Surface a conflict with an ADR instead of overriding it.

## Commands

`uv run harness` runs the app; `uv run pytest -q`, `uv run mypy src tests` and `uv run ruff check .` must all pass before committing (`mypy --strict`, full type hints; ruff rules are in `pyproject.toml`).

## Conventions

- Every validation gate in the plan gets a `pytest` test. Unit tests use fakes; none needs a running Ollama.
- `guard/` never imports `strands` (ADR 0001).
- Every Ollama request goes through the guard client, which sets `num_ctx`; a second path would cause reloads. Workflows use `langgraph` with no `langchain-*` model class (ADR 0004).
- Code a model wrote runs only through the `Sandbox` interface; git and `uv sync` belong to the harness and run outside it (ADR 0005).
- Strands agents are built with `callback_handler=None`; the TUI renders the stream, stdout stays clean.
- The app binds `ctrl+c` with `priority=True` to cancel generation instead of quitting.

## Machine facts

- 24 GB Mac; `sysctl iogpu.wired_limit_mb` is 0 (default), so the Metal limit is unmeasured until M1's preflight measures it.
- Installed models: `gemma4:e4b`, `gemma4:e2b`, `gemma4:12b`, `gemma4:12b-mlx`, `tev1:0.8b`. The 26B MoE, `gpt-oss:20b` and `nomic-embed-text` from the plan are not pulled yet. `gemma4:12b` is too memory-heavy for workflow runs here (swap past 5 GiB, 3–5 tok/s); live checks use `gemma4:e4b`.
- `gemma4:e4b` has sliding-window and shared-KV layers (`/api/show`), so the KV formula overestimates it; calibrate (ADR 0002).
- macOS 27: `sandbox-exec` works (verified 2026-10-09); no container runtime is installed. `llama3.1:8b` is not pulled.
- Ollama 0.35.1; `/api/ps` entries include `context_length`.
- `/api/ps` `size`/`size_vram` are wrong for models with a speculative draft (all `gemma4` here; ollama#17251), so never trust them alone for budgets or spill detection (ADR 0002).
