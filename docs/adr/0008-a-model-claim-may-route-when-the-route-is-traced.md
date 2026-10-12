# A model's claim may route a Workflow, when the route is traced and bounded

Amends ADR 0004. ADR 0004 let only deterministic nodes (pytest, a **Probe**, a schema check, a human) choose where a **Workflow** goes next, and F3 kept to it: a **Dispute** routes only through an **Invariant** check. The F3 control shows what that costs. Haiku's Dispute was correct every time a Behavior went **Stuck** (6/6), but no Invariant covered the test, so nothing could act on it. A stronger model (an **Arbiter**), or the Dispute itself, would have rescued those runs. The user accepts routing on a model's claim, provided every such route can be seen and traced afterwards.

## Decision

A conditional edge may depend on a model's claim when all of these hold:

1. **The claim is a schema field.** A route reads a typed field of a node's structured reply (a `Ruling.verdict`, a `Dispute`). It never parses free text.
2. **Every route is recorded.** Each conditional edge writes a **Route record** to the run log, whatever its basis: `from`, `to`, `basis` (`code`, `model` or `human`), the deciding field and its value, and, for a model basis, the model id, the API `request_id`, the claim's reason and its thinking summary (F9). The run summary counts routes by basis, and the bench report shows them.
3. **Every model-routed edge is bounded.** A counter in the state (for example, at most one arbitration per Behavior) guarantees the loop ends. Past the bound, the edge falls back to its code-based route.
4. **A claim never declares work done.** **Green** still comes only from pytest, and grading still comes only from code (hidden tests, F7). A claim may send work back, sideways, to a stronger model, or to a human. It may not mark a Behavior Green, skip a test, or apply a diff.

## Considered options

- **Keep code-only routing and write more Invariants**: still preferred where an Invariant can express the rule, and the Invariant check still runs first. But not every rule is a property over one call ("a strike in a middle frame"), and a human can't foresee every bad test.
- **Let the implementer's Dispute route directly**: the cheapest option, and the F3 control suggests it would often be right. The Arbiter (F10) is tried first, so that a second, stronger opinion is measured against the Dispute alone. Both can be bench variants.
- **Unbounded routing on claims**: rejected. Two models that disagree could loop until the **Spend cap**, and the trace would show a loop rather than a decision.

## Consequences

- ADR 0004's last line still holds: "The model never decides that work is done."
- `execute.py`'s route functions (the `after_*` methods, wrapped by `gate`) write Route records. A trace can then answer "why did the run go here?" without reading prompts.
- F6's YAML loader may accept a condition on a model field if the edge declares `basis: model` and a bound. Without both, the loader still rejects it.
- The Decision layer (ADR 0003) stays advisory. This ADR covers Workflow edges only, not Approvals.
