# Anthropic is the default backend; the Ollama path is frozen

Builds on ADR 0006. The project's goal changes from "make small local models usable" to "a harness that runs on this machine, with cloud models, for exploring Workflows as graphs". The F1 live check settled it: on the same Plans, `claude-haiku-5-5` matched or beat `gemma4:e4b` on every kata (29/30 against 28/30; 21/24 against 11/24), ran 10–50× faster per Behavior, cost $0.067 for ten runs, and left the Mac cool, where e4b batches ran the GPU at 90–96 °C with swap in use. The Anthropic API therefore becomes the default for both Workflows and the chat agent. OpenRouter and other providers are deferred while Anthropic credit covers the work.

## Considered options

- **Remove the Ollama path** (`guard/`, Calibration, Preflight, the model swap, the watchdog's Spill and Truncation checks): rejected for now. It is built, documented and tested, and removing it costs work without moving the graph work forward.
- **Keep Ollama as an equal backend:** rejected. Every feature would need two live checks, one of them slow and hot, for a backend we no longer intend to use.
- **OpenRouter as a third backend now:** deferred. It would fit behind the same `WorkflowModel` seam, but its structured-output support and cost reporting vary per model and would need checking first. Revisit when cost or model choice calls for it.

## Consequences

- **Frozen** means the Ollama code and its tests stay, and `pytest`, `mypy` and `ruff` keep passing over it. It gets no new features, no live checks and no new support in later features, except where that support is free. ADR 0001's rules still hold: `guard/` imports neither `strands` nor `anthropic`. ADR 0004's rule that Ollama requests go only through the guard client also still holds.
- `[workflow] backend` defaults to `"anthropic"`, and the chat agent gets an Anthropic backend (F1b). Ollama stays reachable through config for offline use.
- **Strands' own Anthropic provider requires `anthropic<1.0`**, and the harness uses 1.x. The chat agent therefore gets a thin Strands model over the 1.x SDK, as ADR 0001 already allowed for Ollama.
- **The Guard watches nothing while no model runs on Ollama.**
  - The status bar shows API usage and cost instead of memory and the Budget.
  - Context accounting uses the API's `usage` and the model's real context window, which is far larger than any local Budget.
- **Thinking blocks constrain compaction.** On current Claude models, Haiku 5.5 included, editing earlier turns can invalidate later thinking blocks, and on some accounts the API rejects the edited history with a 400 ("preserved thinking"). Compaction that rewrites earlier turns has to use a form the API accepts, or rely on server-side compaction. F1b must check this; it must not assume the current compaction works unchanged.
- **Live checks and benchmarks run on `claude-haiku-5-5`**, so the thermal and swap monitoring in the working notes no longer applies to them.
- **Machine facts become history.** The facts about Ollama, Metal and the installed models in AGENTS.md describe the frozen path, not the default one.
