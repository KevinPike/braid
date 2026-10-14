# A Decider may run on an API model; in a Workflow it only picks from a list

The Decision layer used to mean tev1 on Ollama (ADR 0003). It is now a **Decider**: anything that answers typed Choice, Noul and Score questions about a state with probabilities. tev1 is one; the other is Jev (`typesafe/jev-1.13`, alias `~typesafe/jev-latest`) on OpenRouter's Decisions API, `POST /api/alpha/decisions`. Its request and answers have tev1's shape, so `JevDecider` reuses the question and answer types and the Decision log, and imports neither `anthropic` nor `strands`. Input is billed (about a fiftieth of a cent a call), output is free, the window is 32K, and each answer reports `usage.cost`.

## Two tiers

- **Chat: advisory only, as in ADR 0003.** Advice on the Approval prompt (`jev: safe p=0.93 (advisory)`) and a nudge to compact sooner. Nothing runs, or skips an Approval, on advice. Any failure, timeout or the session ceiling leaves "unavailable" or "no advice".
- **Workflow: a Decider picks from a fixed list; code still routes.** *Error triage* (`workflow/triage.py`, `[workflow] triage`) sorts a failed Green run into one of five kinds (`setup_error`, `wrong_value`, `exception`, `hang`, `regression`) and adds the canned note written for that kind to the next implementation prompt. No edge reads the answer (ADR 0004, 0008), so a wrong answer costs one sentence of guidance, and below `triage_min_p` or on any failure there is no note. An answer that *did* pick an edge would be a model basis (ADR 0008) and needs its own record and its own ADR.

## Decisions

- **Opt-in.** `[decide] backend = "auto"` keeps today's behaviour (tev1 on Ollama, off on the API backends). `"openrouter"` turns Jev on for the chat, and `[workflow] triage = true` for Execution runs. Both need `OPENROUTER_API_KEY`; without it the layer is off with a note and nothing fails.
- **Cost.** A chat session may spend `[decide] max_cost_usd` (default $0.05), then the Decider stops. In a run, triage cost counts toward the Spend cap and appears under the role `decider` in the summary.
- **Latency.** `[decide] timeout_s` (default 4, at most 10) bounds every call; advice is awaited before an Approval prompt shows, so a slow Decider must not stall it.
- **Privacy.** Chat advice sends the action (tool name and clipped arguments) to OpenRouter, and triage sends the tail of a pytest run. That is one more place project text goes, which ADR 0009 kept off the chat side; hence opt-in.
- **Answers are not calibrated.** Jev's probabilities are the model's own; tev1's are token probabilities. Neither is a gate.
- **Removed.** The `route` question (chat or tool) had no consumer: nothing acted on its answer. "Done-or-continue" was never built. Both are gone from the code and the glossary.
- **Alpha endpoint.** The path is `/api/alpha/`; if it changes only `decide/jev.py` does.

## Not built yet

Triage in the TUI's `/build` and the MCP driver (only the CLI and bench wire it); triage after Red rejections and at Stuck; a live comparison of triage on and off on the bench sets; the Decider's chat cost in the status bar.
