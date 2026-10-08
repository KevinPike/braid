# The guard has no Strands dependency

`guard/` talks to Ollama through its own client and never imports `strands`, so it can be tested against a bare Ollama and the agent framework can be swapped. The Strands provider is wrapped over the guard client (or replaced by a thin custom provider) rather than the guard wrapping Strands, because Strands' Ollama provider may not expose `prompt_eval_count` or per-request `num_ctx`.
