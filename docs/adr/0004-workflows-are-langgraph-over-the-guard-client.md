# Workflows are LangGraph graphs over the guard client; models write, code checks

Workflows (the Plan loop and the Execution loop) are LangGraph `StateGraph`s whose model nodes call `OllamaClient` directly and ask for one JSON object per call (Ollama's `format`), while every pass/fail decision is made by a deterministic node: pytest, a Probe, a schema check, or a human. LangGraph was picked over Strands' own `strands.multiagent.graph` because it is the shared vocabulary for graphs and ships `interrupt()` and checkpointing, which both loops need for human input and resume; the price is a second framework next to Strands.

## Considered options

- **LangChain chat models (`ChatOllama`) inside the graph**: rejected. They would be a second path to Ollama that does not pin `num_ctx`, so models would Reload and the Guard would not see the calls. Only the `langgraph` package is used; nothing from `langchain-*`.
- **A tool-calling agent per node**: rejected for now. Small local models (`gemma4:e4b`, `llama3.1:8b`) are unreliable at multi-step tool use (see the M4 risk), but reliable at returning one schema-shaped answer. Tools can come back per node once a stronger model is in the lineup.
- **Strands multiagent Graph**: viable and already installed; revisit if LangGraph's typing fights `mypy --strict` badly.

## Consequences

- `workflow/` may import `langgraph` and `strands`; `guard/` still imports neither (ADR 0001).
- The model never decides that work is done. A node that asks a model "did this pass?" is a design bug; run the check instead.
