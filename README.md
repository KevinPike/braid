# braid

A learning-grade local agent harness: a Textual terminal UI over a Strands agent running a local Ollama model, with a runtime guard that keeps context and GPU memory healthy.

The plan, milestones and current status live in [`specs/initial.md`](specs/initial.md). Terms such as Guard, Spill, Truncation and Pinned num_ctx are defined in [`CONTEXT.md`](CONTEXT.md), and design decisions are recorded in [`docs/adr/`](docs/adr/). Known issues are tracked in [`specs/bugs.md`](specs/bugs.md).

## Features

- **Streaming chat TUI.** Markdown rendering while a reply streams, Up/Down prompt history, and a prompt queue: prompts sent during a generation wait their turn.
- **Runtime guard.** Every Ollama request goes through one client that pins `num_ctx` and `keep_alive`. Preflight picks a `num_ctx` that fits the GPU budget, calibrating KV-cache cost per model and stepping down to fallback models when needed. A watchdog polls `/api/ps` and per-call metrics to catch Spill and drift. The status bar shows model, GPU share, memory, tokens per second and keep-alive.
- **Context visibility.** Per-call accounting by category (system prompt, tool schemas, history, tool results, current turn), a context panel with a stacked gauge, and a Truncation detector that raises a banner.
- **Context management.** Sliding-window or summarizing compaction (summaries by a smaller model at a ceiling of `num_ctx`), offloading of large tool results to disk, and a SQLite trim log of what was dropped and why.
- **Tools.** Read-only tools bound to one project folder (read file, list directory, search notes, current time) and an optional shell, off by default.
- **Decision layer.** A small local model (`tev1`) answers advisory questions, such as whether a prompt needs tools or whether to compact early. It never overrides the guard.

## Prerequisites

- macOS with [Ollama](https://ollama.com) 0.35+ running (`ollama serve`, or the menu-bar app)
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`); it fetches Python 3.12 itself
- The models named in `harness.toml`, for example:

  ```sh
  ollama pull gemma4:e4b   # daily driver
  ollama pull gemma4:e2b   # summarizer (optional; falls back to the daily driver)
  ```

## Run it

```sh
uv sync
uv run harness            # reads ./harness.toml
uv run harness other.toml # or point at another config
```

| Key | Action |
|---|---|
| Enter | send the message (on an empty input, resume a paused queue) |
| Up / Down | recall earlier prompts |
| Ctrl-C | cancel the current generation and pause the queue (the app keeps running) |
| Esc | clear the prompt queue |
| Ctrl-Q | quit |

Slash commands: `/model [name]`, `/compact`, `/trims`, `/tools`, `/decisions`, `/exit`.

## Configure

Everything is in `harness.toml`: the Ollama host, the model profile (`model`, `num_ctx`, `keep_alive`, system prompt, fallbacks), the compaction strategy, tool settings, the decision layer and the data directory (`~/.harness` by default, for the trim log, calibration cache and offloaded results). Missing keys fall back to the defaults in `src/harness/config.py`.

## Tests

```sh
uv run pytest -q          # unit tests use fakes; no Ollama needed
uv run mypy src tests     # strict type check

HARNESS_LIVE=1 uv run pytest tests/test_live_recall.py tests/test_live_tools.py   # live gates against Ollama
```

## License

[Apache 2.0](LICENSE)
