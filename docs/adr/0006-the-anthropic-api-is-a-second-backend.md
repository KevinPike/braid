# The Anthropic API is a second backend; Ollama still goes only through the guard client

Amends ADR 0004 and ends the Ollama-only decision. Any model call in the harness may run on Claude through the Anthropic API, chosen per use from config or the model id: both **Workflows** (the Plan loop and the Execution loop) and the chat agent. Every **Ollama** request still goes through the guard client, so **Pinned num_ctx** holds and there is still no second path to Ollama. The reason is measurement and quality: e4b gets 28/30 Green with Claude's Plans but 11/24 with its own, the only stronger local model swaps on this 24 GB Mac, and a Claude row at a few cents a run tells "weak model" apart from "bad Plan" for every later change.

## Considered options

- **Stay Ollama-only**: rejected. `gemma4:12b` swaps on this Mac (3–5 tok/s), and long e4b batches run the GPU at 90–96 °C.
- **Anthropic for Workflows only**: rejected. The chat agent would then be the one place still limited to small local models, for no reason the Guard needs.
- **`claude -p` subprocess per call**: rejected. It has no schema-constrained output, no usage or cost figures, and no per-request control.
- **A LangChain Anthropic chat model in Workflows**: rejected for the same reason as `ChatOllama` in ADR 0004: only `langgraph` is used.
- **Share `GuardedWorkflowModel`'s request code**: rejected. It sends `temperature`, and current Claude models reject non-default sampling values with a 400.

## Consequences

- Workflows call Claude through `AnthropicWorkflowModel`, behind the existing `WorkflowModel` protocol, and the chat agent through Strands' Anthropic provider. `anthropic` is never imported by `guard/`, which stays Ollama-only (ADR 0001); an import-graph test checks this.
- When nothing uses Ollama, there is nothing for the Guard to watch: Preflight, the model swap, **Spill** and **Truncation** checks apply only to Ollama-backed work.
- **Data leaves the machine.** Prompts, conversation history, the Plan, file contents and test and run output are sent to Anthropic. A project the user would not share stays on Ollama.
- **Use costs money.** Every API call records tokens, cost and the API `request_id`, and a **Spend cap** per Workflow run stops it at the next node boundary with nothing applied. The cap's value is config, not part of this decision.
- Credentials come from the SDK's default resolution (`ANTHROPIC_API_KEY`, else an `ant auth` profile). They never appear in `harness.toml`, the run log or the summary, and the **Sandbox** env scrub (ADR 0005) keeps them from code a model wrote.
- Routing is unchanged. A stronger model still never decides that work is done (ADR 0004).
