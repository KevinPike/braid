# Local agent harness

Python 3.12 TUI (Textual) over a Strands agent on local Ollama, with a runtime guard for context and memory health.

## Read first

- **Plan and milestone status**: `specs/initial.md`. Milestones are zero-indexed; M0 is done. Work the next unticked milestone and meet its validation gate before starting another.
- **Vocabulary**: `CONTEXT.md`. Use its terms (Guard, Spill, Truncation, Pinned num_ctx) in code, tests and commits.
- **Why a design is shaped this way**: `docs/adr/`. Surface a conflict with an ADR instead of overriding it.

## Commands

`uv run harness` runs the app; `uv run pytest -q` and `uv run mypy src tests` must both pass before committing (`mypy --strict`, full type hints).

## Conventions

- Every validation gate in the plan gets a `pytest` test. Unit tests use fakes; none needs a running Ollama.
- `guard/` never imports `strands` (ADR 0001).
- Every Ollama request goes through the guard client, which sets `num_ctx`; a second path would cause reloads.
- Strands agents are built with `callback_handler=None`; the TUI renders the stream, stdout stays clean.
- The app binds `ctrl+c` with `priority=True` to cancel generation instead of quitting.

## Machine facts

- 24 GB Mac; `sysctl iogpu.wired_limit_mb` is 0 (default), so the Metal limit is unmeasured until M1's preflight measures it.
- Installed models: `gemma4:e4b`, `gemma4:12b`, `gemma4:12b-mlx`, `tev1:0.8b`. The 26B MoE, `gemma4:e2b`, `gpt-oss:20b` and `nomic-embed-text` from the plan are not pulled yet.
- `gemma4:e4b` has sliding-window and shared-KV layers (`/api/show`), so the KV formula overestimates it; calibrate (ADR 0002).
- Ollama 0.35.1; `/api/ps` entries include `context_length`.
- `/api/ps` `size`/`size_vram` are wrong for models with a speculative draft (all `gemma4` here; ollama#17251), so never trust them alone for budgets or spill detection (ADR 0002).
