# The Execution loop runs in a sandboxed Workspace; Approval moves to the diff

The Execution loop writes into a Workspace (a git worktree of the project) and runs every command there (pytest, Probes) under a Seatbelt Sandbox (`sandbox-exec`) that allows writes only inside the Workspace, blocks reads of the home folder beyond the toolchain, blocks the network, and gets a scrubbed environment. Inside that boundary commands run without a prompt each; Approval is required to apply the Workspace diff to the project and for each Sandbox escape. This amends ADR 0003 for Workspace runs only: the chat agent's `shell` and `write_file` act on the real project and stay gated per call.

Path checks were not enough. On 2026-10-09 a model-style test file run by plain pytest read `~/.zshrc` and opened a network connection; `resolve_inside` never sees code that pytest executes. The same file under a ten-line Seatbelt profile had all three escapes (home read, write outside, network) blocked and the in-Workspace write allowed.

## Considered options

- **Path checks only**: rejected; they guard tool arguments, not what a process does.
- **Approve every command**: rejected; a TDD loop runs pytest dozens of times, and approving each one trains the user to press y without reading.
- **Containers or a VM** (Apple `container`, OrbStack, Lima): stronger, but none is installed, each start costs about a second, they need a Linux toolchain, and their memory comes out of the budget Ollama needs. Kept as a later backend behind the same `Sandbox` interface for runs that must install arbitrary dependencies.

## Consequences

- `sandbox-exec` has been marked deprecated by Apple for years but works on macOS 27 and is what other agent harnesses use; it sits behind a `Sandbox` interface so a container backend can replace it.
- Seatbelt limits access, not resources: the runner enforces timeouts and kills the whole process group, because a runaway test can push Ollama into a Spill.
- Git operations and dependency sync (`uv sync --frozen` from the lockfile at `HEAD`) run in the harness, outside the Sandbox, before the model writes anything. The project's `.git` stays unwritable from the Sandbox, so model code cannot plant a git hook that later runs unsandboxed.
- The diff shown for Approval flags files that run on import or install (`conftest.py`, `sitecustomize.py`, `*.pth`, `pyproject.toml`, lockfiles, CI config), because applying them hands that code to unsandboxed runs. Apply refuses symlinks and paths that leave the project.
