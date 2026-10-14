# braid

**Describe a feature. Get tested code. Pay a few cents.**

braid plans your feature with you, then builds it test-first in a sandbox. A model writes the tests and the code. Code decides whether they pass.

## Models write. Code decides.

A model can say "done" and be wrong. So no step in braid ends on a model's say-so. Every decision is a pytest run, a path rule, an Invariant check or a person.

Two Workflows do the job:

```mermaid
flowchart LR
    G([Your goal]) --> R[Research]
    R --> C[Critiques]
    C --> A{{You approve}}
    A --> P([Plan])
    P --> B[Behavior]
    B --> Red[Red: a failing test]
    Red --> Green[Green: code that passes]
    Green --> S{{You sign off}}
    S --> Pol[Polish: refactor, tests stay green]
    Pol --> D([Diff for your approval])
```

- **Plan loop.** Your goal becomes a Plan: research, Critiques, then your review.
- **Execution loop.** Each Behavior goes Red, then Green. You sign off from a demo. Polish refactors, and braid keeps the result only if the suite stays green.

Runs are checkpointed. Cancel one and it resumes where it stopped.

## Model-written code stays in a box

Code a model wrote runs only inside macOS Seatbelt. It works in a throwaway git worktree, with no network. Its environment is scrubbed, so your API key never reaches it.

The result is a diff. The diff reaches your project only when you approve it.

## You set the price

- A default run on `claude-haiku-5-5` costs a few cents.
- A Spend cap stops a Workflow run at the dollar limit you set (`max_cost_usd`). Benchmarks have their own (`--max-cost`).
- The status bar shows the running cost of your session.

## Quickstart

You need macOS, [`uv`](https://docs.astral.sh/uv/) and an `ANTHROPIC_API_KEY`.

```sh
uv sync
uv run --env-file .env harness
```

Then, in the terminal UI:

```
/plan Add a --json flag to the export command
/build
```

Keys, commands, config and shell usage: [docs/usage.md](docs/usage.md).

## Also in the box

- **Benchmarks.** Run a set of katas N times. `report --compare A B` tells a real change from noise.
- **OpenRouter.** Run a Workflow on another vendor's model. braid refuses models without structured output.
- **Chat TUI.** Streaming Markdown, resumable sessions and a transcript inspector.
- **MCP server.** `braid-mcp` lets Claude Code start runs under spend ceilings. It cannot give a Sign-off or apply a diff.
- **Invariants.** A Plan can state rules. Code checks each test against them.
- **Skills, MCP client and Choice.** Extend the chat agent. Every call that touches your machine needs your approval.

The original local path (Ollama, with a guard for context and GPU memory) is frozen but kept ([ADR 0007](docs/adr/0007-anthropic-is-the-default-backend-ollama-is-frozen.md)).

## Go deeper

- Plans and status: [`specs/features.md`](specs/features.md), [`specs/initial.md`](specs/initial.md)
- Terms such as Workflow, Behavior, Plan and Invariant: [`CONTEXT.md`](CONTEXT.md)
- Why the design looks this way: [`docs/adr/`](docs/adr/)

## License

[Apache 2.0](LICENSE)
