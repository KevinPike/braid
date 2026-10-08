# harness

A learning-grade local agent harness: a Textual terminal UI over a Strands agent running a local Ollama model. See [`specs/initial.md`](specs/initial.md) for the full plan and milestone status (currently M0 — Skeleton).

## Prerequisites

- macOS with [Ollama](https://ollama.com) 0.35+ running (`ollama serve`, or the menu-bar app)
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`); it fetches Python 3.12 itself
- The daily-driver model pulled:

  ```sh
  ollama pull gemma4:e4b
  ```

## Run it

```sh
uv sync
uv run harness            # reads ./harness.toml
uv run harness other.toml # or point at another config
```

| Key | Action |
|---|---|
| Enter | send the message |
| Ctrl-C | cancel the current generation (the app keeps running) |
| Ctrl-Q | quit |

The bottom status bar is intentionally empty until M1 (runtime guard).

## Configure

Edit `harness.toml`:

```toml
[ollama]
host = "http://localhost:11434"

[profile]
model = "gemma4:e4b"
num_ctx = 32768
keep_alive = "30m"
system_prompt = "You are a concise, helpful assistant running locally."

[paths]
data_dir = "~/.harness"
```

## Try the M0 validation gate yourself

1. **10-turn chat streams without blocking the UI.** Start the app and send ten messages in a row. Tokens should appear incrementally, and you should be able to scroll the chat while a reply is streaming. Ask a follow-up like "what did I first ask you?" to confirm history is kept.
2. **Ctrl-C cancels cleanly.** Send "write a 500 word story", press Ctrl-C mid-stream. Generation stops, `[cancelled]` appears, and you can send another message right away. The app must not exit.
3. **Backend errors don't crash the UI.** Stop Ollama (`pkill ollama`), send a message: an `[error: ...]` line appears in the chat pane. Restart Ollama and send again.

To watch what Ollama is doing while you chat: `watch -n1 ollama ps` in another terminal.

## Automated tests

```sh
uv run pytest -q          # config loading, 10-turn streaming, Ctrl-C cancel (fake backend, no Ollama needed)
uv run mypy src tests     # strict type check
```

## Layout

```
harness.toml          model profile and paths
specs/initial.md      plan and milestone tracking
src/harness/
  app.py              Textual app (chat pane, input, status bar placeholder)
  agent.py            Strands Agent on the Ollama provider, text-delta stream
  config.py           harness.toml loading
tests/test_app.py
```
