# Local Agent Harness

A terminal agent harness that runs local Ollama models on a 24 GB Mac and keeps context use and runtime health visible and guarded. It exists to surface what Ollama hides.

## Language

### Runtime health

**Guard**:
The standalone `guard/` module that chooses a safe context length before a model loads and watches every call afterwards. Knows nothing about Strands.
_Avoid_: Monitor, supervisor

**Preflight**:
The loud, fail-fast checklist run before a model loads; ends with a **Budget** check and a confirmed full-GPU load.
_Avoid_: Startup check, health check

**Watchdog**:
The in-session half of the **Guard**: a background `/api/ps` poller plus per-call metric checks, each raising an **Alert**.
_Avoid_: Poller, health monitor

**Alert**:
A watchdog finding with a severity (yellow or red) shown in the status bar or banner.
_Avoid_: Warning, error

**Budget**:
The largest `num_ctx` that fits the GPU limit after margin, helpers, weights and overhead.
_Avoid_: Limit, quota

**Calibration**:
Measuring a model's KV bytes per token by loading it at two context lengths, cached per Ollama version and model digest.
_Avoid_: Estimation, benchmarking

**Pinned num_ctx**:
The single `num_ctx` a model runs at for a whole session, set only by the one client every request goes through.
_Avoid_: Context size, context setting

### Failure modes

**Spill**:
A loaded model with `size_vram < size`, meaning part of it runs on CPU.
_Avoid_: Offload, fallback

**Truncation**:
Ollama silently dropping prompt tokens because the prompt exceeded `num_ctx`; detected when `prompt_eval_count` falls below the harness estimate.
_Avoid_: Overflow, clipping

**Reload**:
Ollama loading a model again mid-session, usually from `num_ctx` drift or eviction.
_Avoid_: Restart

### Context

**Context window**:
The pinned `num_ctx` tokens a call may use; split into categories (system prompt, tool schemas, history, tool results, current turn).
_Avoid_: Prompt size, memory

**Compaction**:
Shrinking history to stay under the pressure thresholds, by sliding window, summary or offloading; every step is recorded in the **Trim log**.
_Avoid_: Truncation (reserved for the failure mode), pruning

**Trim log**:
The SQLite record of what was dropped or summarized, when and why.
_Avoid_: Compaction history

**Prompt queue**:
Prompts submitted while a generation is running; they wait their turn in order and are held while a red **Alert** is active or an approval is pending.
_Avoid_: Backlog, pending messages

**Approval**:
The user's y/n before a gated action runs: a gated tool call (`shell`, `write_file`), applying a **Workspace** diff, or a **Sandbox escape**. The only thing that lets one run. Held open, it holds the **Prompt queue**.
_Avoid_: Confirmation, permission

**Policy check**:
The tev1 question "is this action safe to run in the project folder?", shown on the **Approval** prompt as advice and logged in the Decision log. Never skips the **Approval**.
_Avoid_: Safety check, guardrail

**Hook events**:
Guard, context, decision, session and approval events published on the app's Strands hook registry (`app.hooks`); consumers register there instead of subscribing to components.
_Avoid_: Listeners, callbacks

### Workflows

**Workflow**:
A graph of model steps and deterministic checks that the harness runs toward a goal, pausing for a human where a judgement is needed. The **Plan loop** and the **Execution loop** are the two.
_Avoid_: Pipeline, chain, agent

**Plan loop**:
The **Workflow** that turns a goal into an approved **Plan** through rounds of research, **Critiques** and human review.
_Avoid_: Planning phase, design loop

**Plan**:
The approved hand-off from the **Plan loop** to the **Execution loop**: a goal, its **Behaviors**, its **Assumptions**, what is out of scope, and any open questions.
_Avoid_: Spec, design doc

**Critique**:
One angle's review of a draft **Plan** (feasibility, scope, testability, risk); several run side by side each round.
_Avoid_: Review (the human's turn is the review), feedback

**Behavior**:
One observable thing the feature must do, proven by one test.
_Avoid_: Requirement, story, task

**Assumption**:
A claim the **Plan** depends on, backed by evidence found in research or by a **Probe**.
_Avoid_: Precondition, premise

**Probe**:
A shell command whose exit status confirms an **Assumption**; it runs in the **Sandbox**.
_Avoid_: Check, test

**Execution loop**:
The **Workflow** that runs the **Assumption check**, makes each **Behavior** go **Red** then **Green**, asks for **Sign-off**, then runs **Polish**.
_Avoid_: Build loop, TDD loop

**Assumption check**:
The first stage of the **Execution loop**: run every **Probe** and stop for the human on a failure or an open question, before any code is written.
_Avoid_: Preflight (that is the **Guard**'s checklist), validation

**Red**:
A **Behavior**'s new test failing for the expected reason before the code that satisfies it exists.
_Avoid_: Failing test (a failing test can also mean broken)

**Green**:
The whole test suite passing after an implementation attempt.
_Avoid_: Passing

**Stuck**:
A **Behavior** that is still not **Green** after the attempt limit; the **Execution loop** waits for a human hint or abort.
_Avoid_: Failed, blocked

**Sign-off**:
The human confirming, from a demo, that the working feature does what they expect; the gate between making it work and **Polish**.
_Avoid_: Acceptance (reserved for **Validation gate** language), approval (that applies the diff)

**Polish**:
A behavior-preserving refactor after **Sign-off**, kept only if the suite stays **Green** and reverted otherwise.
_Avoid_: Cleanup, refactor stage

### Isolation

**Workspace**:
The throwaway git worktree an **Execution loop** writes in; its diff reaches the project only through an **Approval**.
_Avoid_: Sandbox (that is the enforcement), scratch dir

**Sandbox**:
OS-enforced limits on a process the harness starts: writes only in its **Workspace** (or the project, for the chat shell), no reads of the home folder beyond the toolchain, no network, a scrubbed environment.
_Avoid_: Jail, container, trusted folder

**Sandbox escape**:
An action the **Sandbox** would block, such as network access or installing a dependency, that the user can allow once through an **Approval**.
_Avoid_: Exception, override

### Models

**Daily driver**:
The model that runs the main loop (`gemma4:e4b`).
_Avoid_: Main model, default model

**Decision layer**:
The tev1 models that answer routing, compaction-trigger, policy and done-or-continue questions. Advisory only; never the sole check on a risky action.
_Avoid_: Router, classifier

**Profile**:
The set of models, roles and starting `num_ctx` values in `harness.toml`.
_Avoid_: Preset, config (config is the whole file)

### Process

**Milestone**:
A numbered, zero-indexed unit of the plan (M0–M15) that ships something usable and has a validation gate that must pass before the next starts.
_Avoid_: Phase, sprint

**Validation gate**:
The checkable criteria that close a **Milestone**.
_Avoid_: Acceptance test
