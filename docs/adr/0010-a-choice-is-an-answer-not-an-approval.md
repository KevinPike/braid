# A Choice is an answer, not an Approval

A **Choice** lets the chat agent, a skill or a **Workflow** node put one to four structured questions to the human mid-run (`specs/ask-user-question.md`). It is modelled on Claude Code's AskUserQuestion, and the tool keeps its name, `ask_user_question`, because models already know it. A Choice looks a lot like an **Approval**: a widget in the chat log, a human pick, the **Prompt queue** held until it is answered. The two must not blur.

## Decision

A Choice's answer goes to the asker as data. It never lets a gated action run, and `ApprovalGate` never reads it (ADR 0003). Gated tools stay gated whatever a Choice answered.

The schema refuses confirmation-shaped Choices: a question whose every option is a yes, no, ok, proceed, continue, cancel, approve, deny, sure or stop is refused with a message the model can act on. That is a deterministic check in the model's validators (ADR 0004), so that answering a Choice does not wear the habit of approving.

A Choice also refuses to ask for a password, key, token or secret, and text shaped like a credential is shown and logged as `[redacted]`.

The two share one human at a time: the app's `_human_lock` is held by whichever widget is open, so a Choice and an Approval from the same turn open one after the other, never together, and neither answers the other. Both hold the Prompt queue.

## Considered options

- **Fold Approvals into Choices, with yes and no options, for one widget.** Rejected. It would let a model write its own Approval prompt, and it would drop the **Policy check** advice and the exact-arguments view that make an Approval a gate (ADR 0003). The Approval's text comes from the harness, from the call's real arguments; a Choice's text comes from the model.
- **Reuse the Decision log for the answers.** Rejected. That log is defined as tev1's advice, and a human's answer is neither tev1's nor advice. Choices have their own log.

## Consequences

- A Workflow node may stop on a Choice and route on the answer (interrupt kind `choice`), and the Route record's basis is `human` (ADR 0004, 0008). This needs no change to either: a human node has always been allowed to route. The answer routes work; it never declares it done.
- The confirmation heuristic is a word list. It will miss paraphrases ("Go ahead", "Hold off"). That is acceptable, because a Choice gates nothing; misses are logged for review.
- Skills depend on the tool name and on the Choice schema, so this is hard to reverse once they do.
