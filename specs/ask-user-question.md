# Choices: structured questions to the human mid-run

Planned 2026-10-13. This spec lets the chat agent, a skill (see `specs/skills-and-mcp.md`, a separate plan) and a **Workflow** node ask the human 1–4 structured questions and get the answers back. It is modelled on Claude Code's AskUserQuestion. Terms are from `CONTEXT.md`. Every gate is a pytest test with fakes: no network, no Ollama, no spend.

## New term (to add to `CONTEXT.md` in C1)

**Choice**:
A set of one to four questions that a model or a Workflow node puts to the human, each with two to four options, one or several to pick, and a free-text Other. The answers go back to the asker. A Choice is for decisions that belong to the user. It gates nothing: it is never an **Approval** and never a **Sign-off**. While it is open it holds the **Prompt queue**.
_Avoid_: Question (taken by the Plan's open questions and by tev1's questions), confirmation, prompt

The tool keeps Claude Code's name, `ask_user_question`, because models already know it.

## Goals

- The chat agent can call `ask_user_question` in the TUI. The answers come back as the tool result, keyed by question text, with optional per-question notes.
- A skill gets the same tool, with no skill-specific code path.
- A Workflow can stop on a Choice (interrupt kind `choice`). An answer that routes writes a Route record with basis `human` (ADR 0004, 0008).
- `harness-workflow` without a TTY reduces a Choice to a numbered stdin prompt.
- Every Choice is logged: the questions, the options, the answer, the latency and the status.
- Offline demo: `uv run harness --demo choice` runs a scripted model that asks a Choice. It needs no key and no network.

## Non-goals

- A Choice never replaces or skips an Approval. Gated tools stay gated whatever a Choice answered (ADR 0003).
- Confirming plans ("Shall I proceed?"). The schema refuses confirmation-shaped Choices (see Validation).
- Asking for secrets. A Choice is refused if it asks for a password, key or token.
- Workflow model nodes calling the tool. ADR 0004 rejected tool-calling agents per node, so a Workflow asks from code, through `interrupt`.
- MCP elicitation. It could later go through the same `Asker` (Open question 5).
- Ollama work. The tool is backend-agnostic, but only the Anthropic path is tested live (ADR 0007).

## The model (`src/harness/choice.py`)

```python
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
_STRICT = ConfigDict(strict=True, extra="forbid", frozen=True, alias_generator=to_camel, populate_by_name=True)

class Option(BaseModel):
    model_config = _STRICT
    label: Label                                                      # "Other" is reserved: it is added automatically
    description: Annotated[str, StringConstraints(max_length=200)] = ""
    preview: Annotated[str, StringConstraints(max_length=2000)] | None = None   # code or ASCII, shown beside the list

class Question(BaseModel):
    model_config = _STRICT
    header: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12)]   # the chip
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    options: Annotated[tuple[Option, ...], Field(min_length=2, max_length=4)]
    multi_select: bool = False                                        # multiSelect on the wire

class ChoiceRequest(BaseModel):
    model_config = _STRICT
    questions: Annotated[tuple[Question, ...], Field(min_length=1, max_length=4)]
    # model validators: unique question texts; unique labels per question; no "Other" label;
    # not confirmation-shaped; no secret asked for or shown (see Validation)

class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    picked: tuple[str, ...] = ()                                      # labels; exactly one unless multi_select
    other: str | None = None                                          # the free text, when Other was picked
    note: str = ""

ChoiceStatus = Literal["answered", "declined", "cancelled", "timed_out", "refused"]

class ChoiceResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    status: ChoiceStatus
    answers: dict[str, Answer] = {}                                   # keyed by Question.question
    reason: str = ""                                                  # why refused or declined

    def for_model(self) -> dict[str, Any]:
        """The tool result: {"answers": {question: "label" | "a, b" | other text}, "notes": {question: note}},
        or {"status": ..., "reason": ...} when it was not answered."""

@dataclass(frozen=True)
class ChoiceRecord:                                                   # one row of the Choice log
    source: str          # "chat" | "skill:<name>" | "workflow:<run_id>"
    model: str           # the model id of the asker, or "code" for a Workflow node
    request: ChoiceRequest
    result: ChoiceResult
    latency_s: float     # from the widget mounting to the answer
    at: datetime | None = None
    id: int | None = None
```

Every type passes `mypy --strict`, and nothing is `Any` except `for_model`'s JSON. The wire schema for the tool is `ChoiceRequest.model_json_schema(by_alias=True)`, so the model sees `multiSelect` as Claude Code spells it.

**Validation** (deterministic code, in the model validators, ADR 0004). A failure raises `ValidationError`, the tool returns `status="refused"` with the message, and the model can retry within the per-turn limit.

| Check | Rule |
|---|---|
| Confirmation | Refused if every label, lower-cased, is in `{yes, no, ok, proceed, continue, cancel, approve, deny, sure, stop}`: that is a confirmation, not a decision |
| Secrets | Refused if a question asks for a password, API key, token or secret. Text matching the gitleaks-style patterns (`sk-…`, `ghp_…`, AWS keys, a high-entropy run of 32 or more characters) is shown and logged as `[redacted]` |
| Size | The field limits above. A preview is clipped to 30 lines for display |
| Uniqueness | Question texts are unique in a request, and labels are unique in a question |

## Interfaces and the module seam

The deep module is `harness/choice.py`. Its interface is small: the types, `validate(raw) -> ChoiceRequest`, `numbered(request) -> str` with `parse_numbered(request, lines) -> ChoiceResult` for stdin, and `ChoiceLog`. Behind it sit validation, redaction, the confirmation check, the numbered-answer parser and SQLite. The **seam** is one callable:

```python
Asker = Callable[[ChoiceRequest, str], Awaitable[ChoiceResult]]   # (request, source) -> result; never raises for a user outcome
```

There are three adapters, so the seam is real: the TUI (`HarnessApp.ask_choice`), stdin (`workflow/cli.py`) and a scripted fake (`tests/choice_fakes.py`, also used by the demo).

| File | Role |
|---|---|
| `src/harness/choice.py` | Types, validation, redaction, numbered render and parse, `ChoiceLog` (SQLite, `[paths] choices_file`) |
| `src/harness/tools/ask.py` | `build_ask_tool(asker, limits) -> AgentTool`: a Strands tool with the hand-written spec above, so the JSON schema is the Pydantic one, not one inferred from a signature. Async. It counts calls per turn |
| `src/harness/chat.py` | `ChoicePrompt(Vertical)`: the inline widget, next to `ApprovalPrompt` and `AnswerPrompt` |
| `src/harness/app.py` | `ask_choice` (the TUI Asker), `_choices_waiting`, `_pump` and the queue label, `action_cancel`, `--demo choice` |
| `src/harness/hooks.py` | `ChoiceRequested` and `ChoiceResolved` **Hook events**. `ChoiceLog` subscribes to them through `app.hooks` |
| `src/harness/workflow_pane.py` | `_ask` handles `kind == "choice"` by building a `ChoiceRequest` and awaiting `app.ask_choice` |
| `src/harness/workflow/cli.py` | `_stdin` handles `kind == "choice"` with `numbered` and `parse_numbered` |
| `src/harness/demo.py` | A scripted Anthropic transport (`httpx2.MockTransport`) for `--demo choice`, moved out of `tests/anthropic_fakes.py` |

The tool is registered like the other built-ins in `_tools()`, behind `[tools] ask = true` (default on). It needs no Approval, because it acts on nothing. `ApprovalGate` does not gate it, and a Choice answer is never read by `ApprovalGate`.

**Skills**: a skill's instructions tell the chat agent to call `ask_user_question`, and the tool is the same one. When the skills plan tracks the active skill, the Asker's `source` becomes `skill:<name>`. That is the only skill-specific part. A skill that lists `allowed-tools` without `ask_user_question` does not get it.

## UI design

The widget is inline in the chat log, not a `ModalScreen`, to match `ApprovalPrompt` and `AnswerPrompt`: the chat stays visible, and the app's `ctrl+c` (`priority=True`) still fires. It takes focus when mounted and gives it back when it collapses (`_collapse`).

```
╭─ Choice from claude-haiku-5-5 ──────────────────────────────────────────────────────╮
│ [Hashing ✓] [Storage]                               question 2/2                     │
│ Where should the kata keep its users?                                                │
│                                                  │ preview                           │
│ ❯ 1 SQLite      one file, stdlib sqlite3         │ CREATE TABLE users (              │
│   2 JSON file   simplest, no queries             │   id INTEGER PRIMARY KEY,         │
│   3 In memory   lost on exit; tests only         │   name TEXT NOT NULL              │
│   4 Other…      type your own answer             │ );                                │
│ note (optional): ____________________________                                        │
│ 1-4 pick · ↑↓ move · space toggle (multi) · enter next · ⇧tab back · esc decline     │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

- The header chips show progress (`✓` when answered). Enter on the last question submits the whole Choice. Tab and Shift-Tab move between questions, and answers are kept.
- Other opens an `Input` in place, and submitting it empty is rejected. `n` focuses the note field.
- The preview pane is shown only when the highlighted option has a preview. Under 100 columns it stacks below the list. It is `markup=False`, like every other widget that shows model text.
- Once answered, the widget collapses to one record line: `✓ Hashing? argon2-cffi · Storage? SQLite (6.1 s)`. Declined: `✗ choice declined`. Cancelled: `✗ choice cancelled`.

## Behavior tables

**With the rest of the app**

| Situation | Behavior |
|---|---|
| Prompts queued while a Choice is open | Held. `_pump` adds `self._choices_waiting`, and the label says `held by a choice` |
| An Approval is pending (same turn, concurrent tool calls) | One human widget at a time: `_approval_lock` becomes `_human_lock`, shared by `ask_approval` and `ask_choice`, so they queue in call order. Neither answers the other |
| Esc in the widget | `status="declined"`. The model gets `{"status": "declined"}` and the turn goes on |
| Ctrl+C while a Choice is open in chat | `action_cancel` resolves every chat `ChoicePrompt` as `cancelled` (as it denies Approvals), then cancels the worker. `settle` drops the unanswered tool use. Logged as `cancelled` |
| The model asks after Ctrl+C (a race at the end of the stream) | `ask_choice` checks that the caller's `run_id` is the current one and `generating` is true. If not, it returns `cancelled` without mounting anything |
| A Workflow is waiting on a human | A chat generation cannot run (the queue is held), so the chat agent cannot ask. A Workflow Choice counts in `_pauses_waiting` like `AnswerPrompt` |
| Ctrl+C while a Workflow Choice is open | As for the other interrupts: the run pauses at its last Checkpoint, and the Choice is asked again on resume. Logged as `cancelled` |
| Session switch or quit with a Choice open | `cancelled`. Nothing is saved as answered |

**Limits** (`[ask]` in `harness.toml`)

| Limit | Default | When it is passed |
|---|---|---|
| Questions per call | 4 (schema) | `refused` |
| Calls per user turn | 3 | `refused`, reason `"ask limit for this turn reached; decide or stop"` |
| Answer timeout | 0, meaning wait as an Approval does | `timed_out`. No option is chosen for the user |
| Refusals before the tool stops for the turn | 2 | Later calls are `refused` without being shown |

**Headless (`harness-workflow`, stdin)**

```
[choice 1/2] Storage — Where should the kata keep its users?
  1) SQLite — one file, stdlib sqlite3
  2) JSON file — simplest, no queries
  3) Other (type: o <text>)
  > 1            (multi: 1,3 · Other: o argon2 via cffi · note: 1 # because tests)
```

Previews are printed indented under their option. Bad input is asked again, at most 3 times, and then the answer is `declined`. EOF on stdin is `cancelled`, and the run pauses at its Checkpoint (`CANCELLED_EXIT`). With `--no-hints` there is no human: the Choice is answered by code (Route basis `code`, as `answered()` already does), with the option the node marked `default`. A node with no default raises, as `cli.py:242` does for other unexpected interrupts.

**Routing (ADR 0004, 0008)**

A Workflow node raises `interrupt({"kind": "choice", "request": req.model_dump(by_alias=True)})`. The answer goes back into the state as `ChoiceResult`, and the `after_*` function returns `Route(to, "choice", {header: picked}, self.answered())`, so the basis is `human` (`code` in a no-hints run). This needs no ADR change: a human node has always been allowed to route. If a model proposes the Choice inside a structured reply (a `Choice` schema field), ADR 0008 applies: sending work to a human is allowed, it must be bounded (at most one model-proposed Choice per Behavior), and the record carries the claim. The human's answer still routes with basis `human`.

**Logging**

The `ChoiceLog` row holds source, model, the request (redacted), the result (Other text redacted), status, latency and time. `/choices` lists the last 20 rows, like `/decisions`. A Workflow Choice also writes a `choice` line to the run log. The Choice log is kept separate from the Decision log, because that log is defined as tev1's advice. A human answer is neither tev1's nor advice.

## Milestones

**C1 — The Choice model.** `choice.py` types, validation and redaction, `numbered` and `parse_numbered`. Add the `CONTEXT.md` entry and ADR 0010 (below).
*Gate* (`tests/test_choice.py`): valid requests parse with camelCase and snake_case keys. Each row of the Validation table rejects a crafted request with a message naming the field. A planted `sk-ant-…` string in a preview comes out `[redacted]`. `parse_numbered` handles `1`, `1,3`, `o text`, `2 # note`, and bad input then EOF. `for_model` has the exact shape for single, multi and Other answers.

**C2 — The widget.** `ChoicePrompt`, plus `ask_choice` as the TUI Asker with the shared `_human_lock`.
*Gate* (`tests/test_choice_widget.py`, `app.run_test()` pilot with a fake replier): pressing `2` then Enter returns that label. Space toggles in multi-select. Other plus typed text returns `other`. `n` adds a note. Esc gives `declined`. The preview pane appears only for an option that has a preview. The record line replaces the widget, and focus goes back to `#prompt`. A queued prompt does not start until the Choice is answered.

**C3 — The chat tool and the offline demo.** `tools/ask.py`, registered in `_tools()`, `--demo choice`.
*Gate* (`tests/test_ask_tool.py`, `FakeApi` over `httpx2.MockTransport`): a scripted `tool_use` for `ask_user_question` mounts the widget. The pilot answers. The next request carries a `tool_result` equal to `for_model()`. The fourth call in a turn is `refused`. A Ctrl+C while the widget is open leaves no unpaired tool use in history, and nothing is answered. With `shell` gated and called in the same turn, the Approval and the Choice open one after the other, never both. `--demo choice` boots with no `ANTHROPIC_API_KEY` set and makes no real request (the transport would fail).

**C4 — The log and the limits.** `ChoiceLog`, Hook events, `/choices`, `[ask]` config, timeout.
*Gate*: each status (answered, declined, cancelled, timed_out, refused) writes one row with latency, and a fake clock makes the latency exact. A secret typed in Other is redacted in the row but not in the tool result. A timeout of 0.1 s gives `timed_out` and picks nothing.

**C5 — Workflows and stdin.** Interrupt kind `choice` in `workflow_pane._ask` and `cli._stdin`. Pick one node to convert first (Open question 3).
*Gate* (`tests/workflow/test_choice_interrupt.py`, scripted model): a graph with a Choice node routes on the answer and writes a Route record with `basis: human`, `field: choice`. In `--no-hints` mode it writes `basis: code` with the default, and a node with no default raises. Through `main(argv, ask=fake_stdin)`, the numbered prompt is printed, and EOF exits with `CANCELLED_EXIT` and the run left paused. Resuming asks the same Choice again.

**C6 — Skills.** After the skills plan has an active-skill notion: `source="skill:<name>"`, and `allowed-tools` respected.
*Gate*: a fake skill that tells the agent to ask writes a row with `source == "skill:demo"`. A skill whose `allowed-tools` leaves out the tool gets a tool list without it.

**Live check** (after C3, only with the user's approval of spend, `uv run --env-file .env`): five chat prompts on `claude-haiku-5-5` that do and do not call for a decision. Record how often it asks, whether it ever asks to confirm, and the cost. This is measured, not gated.

## ADR to write in C1

**0010: A Choice is an answer, not an Approval.** A Choice's answer goes to the model as data. It never lets a gated action run, and `ApprovalGate` never reads it. The schema refuses confirmation-shaped Choices, so that answering a Choice does not wear the habit of approving. The option considered was to fold Approvals into Choices with yes and no options, for one widget. It was rejected because it would let a model write its own Approval prompt and drop the Policy check advice and the exact-arguments view (ADR 0003). This meets the bar: it is hard to reverse once skills depend on it, surprising without the reason, and a real trade-off.

## Risks

- **Over-asking.** Haiku may ask instead of deciding. The tool description says to ask only about the user's own preferences and decisions, and the per-turn limit and the live check measure it.
- **The confirmation heuristic.** It is a word list, so it will miss paraphrases ("Go ahead" or "Hold off"). That is acceptable, because a Choice gates nothing. Misses are logged for review.
- **Strands concurrent tool execution.** Two `ask_user_question` calls in one turn line up behind `_human_lock`, but the second may be stale after the first answer. Cap at one Choice per assistant message (refuse the extra ones), as Claude Code effectively does.
- **Strands tool spec.** A hand-written `AgentTool` spec must match what the SDK's Anthropic formatter sends. A C3 gate asserts the request's `tools[]` entry.
- **Textual focus and Ctrl+C.** A nested `Input` (Other or note) must not swallow Esc or Ctrl+C. The gates press both while the `Input` has focus.
- **Redaction false positives** on long hashes in previews. Accepted: only display and the log are affected, and the model keeps its own text.

## Open questions (with recommendations)

1. **Name of the term.** Choice, or Ask, or User question? *Recommend **Choice**.* `Ask` is already the interrupt-answer type in `execute.py`, and Question collides with Plan open questions and tev1 questions.
2. **Where the log lives.** A new `ChoiceLog`, or a `kind="choice"` row in the Decision log? *Recommend a separate log*, because the Decision log's meaning is tev1 advice. Revisit if `/decisions` should interleave the two.
3. **Which Workflow node converts first.** `clarify` (Plan open questions become a Choice when the Plan gives options), or Stuck (options: hint, skip, abort, plus Other for a hint)? *Recommend Stuck.* Its answers already route (`is_abort`), and a Choice turns free-text parsing into a typed field. Sign-off stays y/n: it is a Sign-off, not a Choice.
4. **Timeout default.** *Recommend 0 (wait)*, to match Approvals. A timeout that let the model go on would act without the user.
5. **MCP elicitation.** Map `elicitation/create` onto the same Asker? *Recommend yes later, in the skills plan.* It reuses the widget and log, with `source="mcp:<server>"`, and it needs no change here.
6. **ADR number.** *Settled 2026-10-13: the Choice ADR is 0010; the skills and MCP ADRs are 0011-0013.*
7. **Answer format to the model.** JSON, or Claude Code's prose (`"Q"="A"`)? *Recommend JSON from `for_model()`*. It is typed, testable, and Haiku reads it fine. The live check confirms it.
