# Workflows: Spike, Sandbox, Execution loop, Plan loop, TUI (M11–M15)

Two **Workflows** on LangGraph: a **Plan loop** that turns a goal into an approved **Plan**, and an **Execution loop** that checks the Plan's **Assumptions**, makes each **Behavior** go **Red** then **Green**, gets **Sign-off**, then runs **Polish**. Code the model writes runs only inside a **Sandbox**, in a **Workspace**. Terms are defined in `CONTEXT.md`; the decisions behind them are ADR 0004 (LangGraph over the guard client; models write, code checks) and ADR 0005 (sandboxed Workspace; Approval moves to the diff).

## Why, and what success means

braid is for learning, and the learning comes from using it on real hobby work. So:

- **Success** is using the loops on the practice ladder below and being able to change them: add a node, a route or a Critique angle without reading LangGraph docs. Each milestone ends with a **Try it** for the user; agents skip it.
- **Measured, not gated:** how well local models execute. M11 sets the baseline on five katas (Behaviors Green without a hint, attempts per Behavior, minutes per Behavior, Red rejections); each later live check reruns the same katas against it. A weak result changes which model runs and how small the Behaviors are, not whether to build.
- **Gated by tests:** the loop's own promises. No test passes at Red, a Polish that breaks the suite is reverted, nothing leaves the Sandbox, nothing reaches the project without Approval.

**Plans come from Claude by default.** `claude -p` writes Plan JSON from a kata's `GOAL.md` (M11), and the Execution loop runs it locally. The Plan schema is the seam, so the local Plan loop (M14) is built for learning, and its live check compares local Plans with Claude's on the same katas.

**Practice ladder.** Work up in order, in `~/projects/braid-practice/`; this repository is the last rung.

1. Katas (from M11): roman numerals, bowling score, word wrap, semver compare, cron next-run.
2. Small CLIs (from M13): a `todo.txt` CLI, CSV to Markdown table, git-log stats per author, a recipe scaler with unit conversion, and a dotfiles symlink manager as the Sandbox stress test.
3. A feature in an existing hobby repo (from M13): a new flag or subcommand, which tests `Plan.files` and working in code the model did not write.
4. braid itself (after M15): `/export` via `export_markdown`, items from `specs/bugs.md`.

`examples/graphs/` holds runnable versions of both loops with a scripted model. Read the one your milestone builds before writing code: node names, routing and interrupt handling carry over; the `model_*` functions, `sh()` and the temp-dir workspace are what each milestone replaces.

```sh
uv run --no-project --with langgraph python examples/graphs/plan_graph.py
uv run --no-project --with langgraph --with pytest python examples/graphs/exec_graph.py
```

## Working a milestone

1. Read the milestone's section here and every ADR it names. Done when you can say which existing module each new file talks to.
2. Build the tasks in order. Each task is done when its gate tests pass and `uv run mypy src tests` is clean; commit per task.
3. Close the milestone: every gate listed passes, `uv run pytest -q` passes, the checkbox in `specs/initial.md` is ticked, and a dated **Status notes** list under it records what was built, which test proves each gate, and anything that diverged from this spec.
4. When a gate cannot be met as written, or the code pushes against an ADR, record it as an open question in the status notes and stop that task. Keep the gate as written.

**Order:** M11 → M12 → M13 → M14 → M15. M12 uses none of M11's code, so it can run alongside M11; M14 can start once M13's task 13.1 has landed.

**Shared rules**

- Unit tests fake Ollama with `respx` the way `tests/test_guarded_model.py` does; sandbox tests run the real `sandbox-exec` and skip when `sys.platform != "darwin"`.
- Gate tests from a fixture project: build a small git repo in `tmp_path` per test (a `pyproject.toml`, one module, a `tests/` folder); do not touch this repository's tree.
- Every subprocess the workflow code starts goes through the `Sandbox` interface, including in tests (use the unsandboxed `LocalSandbox` from 12.1 there). Git and `uv sync` are the two exceptions, run by `Workspace` itself.
- Workflow events reach the TUI and telemetry through `app.hooks` (M15). Before M15, the CLI prints them.

---

## M11 — Spike: real models in the example Execution loop

Measures whether local models can execute Claude-written Plans, before M12–M15 build the real thing, and gives the user something to run this week. Everything except 11.1 is throwaway code in `examples/graphs/`, held to "runs and reports honestly", not to `mypy --strict`. ADRs: 0004, 0005.

### 11.1 `format` on the guard client

`OllamaClient.chat` gets `format: str | Mapping[str, Any] | None = None`, sent as the body's `format` key when set. Gate (`tests/guard/test_ollama_client.py`): the body carries the schema, and `num_ctx` is still pinned.

### 11.2 Practice repos

Create `~/projects/braid-practice/` holding five kata repos: `roman-numerals`, `bowling`, `word-wrap`, `semver-compare`, `cron-next`. Each one is its own git repo with one commit and contains:

- `pyproject.toml` (Python ≥ 3.12, `pytest` as a dev dependency) and a gitignored `.venv` made by `uv sync`;
- one flat module, `<name>.py`, holding only a docstring;
- an empty `tests/` directory;
- `GOAL.md`: one paragraph saying what the function does, its signature, and two worked examples. A human should be able to check it at a glance.

Done when `uv run pytest` in each repo exits 5 (no tests collected).

### 11.3 Plans from Claude

- `examples/graphs/plan_prompt.md` is the prompt for `claude -p`. It takes `GOAL.md` and `git ls-files` and asks for only a JSON object in the Plan shape that 13.1 later formalises: `goal`, `behaviors`, `assumptions` (`text`, `probe`, `evidence`), `files`, `demo`, `out_of_scope`, `open_questions`. Rules it must state:
  - **Size Behaviors for a small model:** 3–6 Behaviors, simplest first, each provable by one test and about 30 lines of implementation.
  - Every Assumption gets a `probe` that runs in the repo without network access.
  - `demo` is a `python -c` command that prints the function's output on the `GOAL.md` examples.
- `examples/graphs/claude_plan.sh <kata-dir>` runs `claude -p` with that prompt, pulls out the JSON, checks the required keys with a short Python snippet, and writes `<kata-dir>/plan.json` (gitignored in the kata).

Done when the script has produced a valid `plan.json` for one kata. The user produces and reviews the rest.

### 11.4 `examples/graphs/exec_live.py`

This is `exec_graph.py` with the scripted parts replaced.

- **Model:** every `model_*` call goes through `OllamaClient.chat(format=…)`, with the model preflighted and pinned the way `evals/runner.py` does it. Replies use the `FileEdit`/`Edits` schemas from 13.2, and every path passes `resolve_inside`.
- **Workspace:** copy the kata into a temp dir without `.git` and `.venv`, then run `uv sync --frozen` there, outside the Seatbelt profile.
- **Sandbox:** every pytest run, Probe and demo runs under the Seatbelt profile described in 12.1. The `file-read-metadata` line is required; without it Python cannot start.
- **Flags:**
  - `--plan` and `--model`.
  - `--no-hints`: a Stuck Behavior is marked failed and the run moves to the next one; `clarify` waives and Sign-off answers yes. This is the baseline mode.
  - Without `--no-hints`, every interrupt reads its answer from stdin.
- **Summary:** each run writes `<kata>/runs/<timestamp>-<model>.json` with the model, the kata, and per Behavior: whether it reached Green, attempts, Red rejections, hints and seconds. It also records totals, Stuck count, whether Polish was kept or reverted, and the node trace.
- **Report:** `examples/graphs/report.py <practice-dir>` prints a Markdown table of models × katas from those summaries.

**M11 validation:** the 11.1 gate passes, and a `report.py` table from `--no-hints` runs of `gemma4:e4b` and `gemma4:12b` over all five katas is pasted into this milestone's status notes in `specs/initial.md`. The table is a measurement, so it is recorded rather than asserted by pytest, the one exception to "every gate gets a test".

**Try it:** run `claude_plan.sh` on each kata and edit each Plan before running it, noting what you changed. Run one kata interactively and hint when it gets Stuck. Then run the `--no-hints` baseline. Read one trace and find the node where a run went wrong.

---

## M12 — Sandbox and Workspace

Gives every later milestone a safe place to run model-written code, and closes the open item "the shell tool is a working directory, not isolation". No LangGraph yet. ADRs: 0005, 0003.

### 12.1 `sandbox/`: the `Sandbox` interface and Seatbelt backend

Files: `src/harness/sandbox/__init__.py`, `sandbox/policy.py`, `sandbox/seatbelt.py`.

```python
@dataclass(frozen=True)
class SandboxPolicy:
    writable: tuple[Path, ...]          # Workspace (or project root) plus its private tmp dir
    readable: tuple[Path, ...]          # toolchain roots, re-allowed inside denied_read
    denied_read: tuple[Path, ...]       # default (Path.home(),)
    network: bool = False

@dataclass(frozen=True)
class RunResult:
    exit_code: int
    output: str                         # stdout+stderr interleaved, tail-trimmed to max_output_bytes
    timed_out: bool
    duration_s: float

class Sandbox(Protocol):
    async def run(self, command: str, *, cwd: Path, policy: SandboxPolicy,
                  timeout_s: float, env: Mapping[str, str] | None = None) -> RunResult: ...
```

- `SeatbeltSandbox` renders the policy into an SBPL profile and runs `sandbox-exec -f <profile> /bin/sh -c <command>`. `LocalSandbox` runs the same command without `sandbox-exec`; it exists for graph tests and is never selected by config.
- Profile shape, proven on this machine on 2026-10-09: `(allow default)`, `(deny network*)`, `(deny file-write*)` then allow each writable path plus `/dev/null` and `/dev/tty`, `(deny file-read* …denied_read)` then allow each readable and writable path and the literal home dir, and `(allow file-read-metadata)`. Without that last line Python cannot start (`execvp … Operation not permitted`).
- Resolve every path with `Path.resolve()` before rendering; Seatbelt matches real paths (`/tmp` is `/private/tmp`). Escape `"` and `\` in paths.
- Network off means all of `network*`, unix sockets included: the ssh-agent socket is an escape route.
- Environment: start empty and pass `PATH`, `HOME`, `LANG`, `TERM`, plus the caller's `env`. Set `TMPDIR` to a `.tmp` dir inside the first writable path, and do not allow `/private/var/folders`.
- Timeout: start with `start_new_session=True`; on timeout send SIGTERM and then SIGKILL to the process group. Defaults: 120 s, `max_output_bytes` 64 KiB with a `[… trimmed N bytes …]` marker at the head.
- `toolchain_paths(venv: Path) -> tuple[Path, ...]` returns what a project's `.venv` needs to start Python: the venv itself and `sys.base_prefix` of `.venv/bin/python` (resolve the symlink; for uv it lives under `~/.local/share/uv/python/`).

Gates (`tests/test_sandbox.py`). Each puts a secret file in a `denied_read` tmp dir and runs a `python -c` snippet:

1. Reading the secret fails; reading a file in `writable` succeeds.
2. Writing outside `writable` fails and leaves no file; writing inside succeeds.
3. Connecting to a TCP listener the test opened on `127.0.0.1` fails with `network=False` and succeeds with `network=True`. This doubles as proof that model code cannot reach Ollama.
4. An env var set in the test process is invisible to the command unless it is passed in `env`.
5. A command that starts a sleeping grandchild is killed at the timeout, the grandchild included, and returns `timed_out=True`.
6. `python -m pytest --version` from a fixture venv runs under a policy built with `toolchain_paths`.

### 12.2 `sandbox/workspace.py`: Workspace lifecycle

```python
class Workspace:
    path: Path
    project: Path
    @classmethod
    async def create(cls, project: Path, base: Path, run_id: str) -> Workspace: ...
    async def sync_deps(self) -> RunResult: ...
    def policy(self) -> SandboxPolicy: ...
    async def changes(self) -> list[Change]: ...
    async def diff(self) -> str: ...
    async def apply(self) -> None: ...
    async def remove(self) -> None: ...

@dataclass(frozen=True)
class Change:
    path: str
    status: Literal["added", "modified", "deleted"]
    flags: frozenset[Literal["symlink", "outside", "runs-on-import"]]
```

- `create`: `git worktree add --detach <base>/<run_id> HEAD`, with `base` defaulting to `~/.braid/workspaces`. Refuse when `project` is not a git repo. When the project tree is dirty, still create the worktree and return a warning: the Workspace starts from `HEAD`.
- `sync_deps`: when `uv.lock` exists, run `uv sync --frozen` in the Workspace, outside the Sandbox (ADR 0005). Otherwise do nothing.
- `policy()`: writable is the Workspace and its `.tmp`; readable is `toolchain_paths(.venv)`; network is off. The project's `.git` sits under the denied home dir, so model code cannot write hooks. Git commands inside the Sandbox will fail; that is expected.
- `changes` and `diff`: run `git add -A --intent-to-add` and then `git diff HEAD` in the harness, so new files appear. `runs-on-import` flags `conftest.py`, `sitecustomize.py`, `*.pth`, `pyproject.toml`, `setup.py`, `setup.cfg`, `uv.lock`, `.github/**` and `Makefile`.
- `apply`: refuse when any change is flagged `symlink` or `outside`, or when a target file in the project differs from `HEAD`. Otherwise run `git apply` with the Workspace diff in the project. Never auto-commit.
- `remove`: `git worktree remove --force` and delete the directory.

Gates (`tests/test_workspace.py`, with a fixture git repo per test):

1. A new file and an edited file in the Workspace show up in `changes()` and `diff()`; the project tree is untouched until `apply()`, and matches the Workspace after it.
2. A symlink in the Workspace pointing outside makes `apply()` raise, and the project is unchanged.
3. An edited `conftest.py` carries the `runs-on-import` flag.
4. Under `SeatbeltSandbox` with `policy()`, writing `<project>/.git/hooks/pre-commit` from inside the Workspace fails.
5. `remove()` leaves no worktree in `git worktree list`.

### 12.3 Chat shell under the Sandbox

`ProjectSandbox` in `tools/builtin.py` runs each command through `SeatbeltSandbox` with writable = (project root, its `.tmp`) and readable = `toolchain_paths(root/.venv)` when that exists. Network comes from a new `[tools] shell_network = false`. Approval stays per command (ADR 0003). Rewrite the class docstring to describe the new boundary.

Gate (`tests/test_tools.py`): an approved shell command that writes outside the root fails and creates no file.

**M12 validation:** model-style code run in a Workspace cannot read the home folder, write outside the Workspace, reach the network, or outlive its timeout; a Workspace diff reaches the project only through `apply()`, which refuses symlinks and escaping paths.

**Try it:** in the TUI, ask the chat agent to run `curl https://example.com` and `cat ~/.zshrc`, approve both, and watch them fail. Then write your own escape attempt as a test file in a practice repo and run it through `SeatbeltSandbox` from a Python prompt.

---

## M13 — Execution loop (CLI)

The Execution loop from `examples/graphs/exec_graph.py`, with a real model through the guard client, a real Workspace and the Sandbox. CLI only; the TUI is M15. ADRs: 0004, 0005, 0001.

### 13.1 Structured model calls and the Plan schema

- Model calls use the `format` parameter from 11.1.
- `src/harness/workflow/plan.py` holds the Plan hand-off as Pydantic models. `Plan.load(path)` and `Plan.save(path)` work with JSON. Plans live in `~/.braid/plans/<id>.json`.

  ```python
  class Assumption(BaseModel):
      text: str
      probe: str | None = None        # shell command, run in the Sandbox
      evidence: str | None = None     # "path:line — what it shows", from Plan-loop research
  class Plan(BaseModel):
      goal: str
      behaviors: list[str]            # min length 1
      assumptions: list[Assumption]
      files: list[str] = []           # paths the work will read or touch; seeds model context
      demo: str | None = None         # command whose output is shown at Sign-off
      out_of_scope: list[str] = []
      open_questions: list[str] = []
      notes: list[str] = []
  ```

- `src/harness/workflow/model.py`:

  ```python
  class WorkflowModel(Protocol):
      async def complete(self, system: str, prompt: str, schema: type[T]) -> T: ...  # T bound to BaseModel
  ```

  `GuardedWorkflowModel(client: OllamaClient, model_id: str)` sends `format=schema.model_json_schema()`, joins the streamed text, and validates with `schema.model_validate_json`. On a validation error it retries once, appending the error to the prompt, then raises `WorkflowModelError`. Thinking text is dropped.
- Gates (`tests/workflow/test_model.py`, respx): a valid reply parses; one invalid reply followed by a valid one parses after exactly two requests; two invalid replies raise; every request carries `format` and the pinned `num_ctx`.

### 13.2 The graph: `workflow/execute.py`

`build_execution_graph(model: WorkflowModel, sandbox: Sandbox, workspace: Workspace, settings: WorkflowConfig) -> StateGraph` with nodes `check_assumptions`, `clarify`, `write_test`, `run_red`, `check_satisfied`, `write_impl`, `run_green`, `stuck`, `next_behavior`, `sign_off`, `polish`, `verify_polish`, `finish`. Routing is the example's, plus `finish`. Prompts live in `workflow/prompts.py`; model replies use these schemas:

```python
class FileEdit(BaseModel):
    path: str                         # relative to the Workspace
    content: str                      # full new file contents
class Edits(BaseModel):
    edits: list[FileEdit]             # 1–3 files
    rationale: str                    # one sentence, logged
```

Rules each node must keep:

- **Paths:** every `FileEdit.path` goes through `resolve_inside(workspace.path, …)`. A refused edit is not written and counts as a failed attempt, with the refusal in `last_run`.
- **Model context:** the Plan, the current Behavior, the Workspace file tree (`git ls-files`, up to 200 lines), the current contents of `plan.files` plus every file edited so far (up to 300 lines each, with a trim marker), and the tail of the last run (up to 40 lines). Never send the run history.
- **Red** requires pytest exit 1, or exit 2 with `ModuleNotFoundError` or `ImportError` naming a module that is not the test file. A `SyntaxError` or other collection error in the test file routes back to `write_test` with that output. After `max_attempts` (default 3) rejected tests, route to `stuck`.
- **Satisfied** (divergence from the first draft of this spec, decided with the user after M11): a new test can pass at Red because code written for earlier Behaviors already implements it. When `run_red` finds the test passing, the deterministic node `check_satisfied` runs the same test file(s) against the implementation as it was at Workspace creation (HEAD): it temporarily restores every non-test file changed since HEAD (files under `tests/` stay) to its HEAD contents and removes files added outside `tests/`, runs the test, and restores the current contents in a `finally`. If the test fails there (any non-zero exit, import errors included), the Behavior is **Satisfied**: it is recorded (`BehaviorStats.satisfied`, the summary's `satisfied` count) and the run goes to `next_behavior`. If it still passes at HEAD the test is vacuous: it counts as a Red rejection and routes to `write_test` with feedback that it passes without any implementation; after `max_attempts` such rewrites, route to `stuck`. The model is never asked.
- **Green** runs the whole suite: `settings.test_command`, default `.venv/bin/python -m pytest -q --no-header -p no:cacheprovider`.
- **Assumption check** runs each `probe` in the Sandbox with a 30 s timeout. Assumptions with `evidence` and no probe pass. A failure or any `open_questions` interrupts at `clarify`. The answer `abort` ends the run; any other answer waives the failed Assumptions, appends the answer to `plan.notes`, clears `open_questions`, and runs the Assumption check again.
- **Stuck** interrupts with the Behavior and the last run. A hint resets attempts and goes into the next `write_impl` prompt; `abort` ends the run.
- **Sign-off** runs `plan.demo` in the Sandbox when set, and shows its output with the list of passing tests. `yes` goes to `polish`; `abort` ends the run; any other text is appended as a new Behavior and routes to `write_test`.
- **Polish** snapshots every file it will change, asks for `Edits` with "change structure and names, never behavior", then runs the suite and each of `settings.lint_commands` (default empty). Red reverts every snapshot. The run gets `max_polish_rounds` (default 1).
- **Finish** interrupts with `workspace.changes()` (flags shown) and the diff. `yes` runs `workspace.apply()` and then `remove()`. Anything else keeps the Workspace and prints its path.
- Each node appends one JSON line to `<runs_dir>/<run_id>.jsonl` (`[workflow] runs_dir`, default `~/.braid/runs`), rich enough to debug a failed Behavior without rerunning: node, timestamp, Behavior index, attempt, and where relevant the prompt sent, the model's parsed reply, files written and refusals, exit code, `timed_out`, the output tail (40 lines), seconds and model latency.

Gates (`tests/workflow/test_execute.py`, using a `ScriptedModel` fake for `WorkflowModel`, `LocalSandbox`, a fixture repo with a venv made by `uv venv` and pytest installed, and scripted interrupt answers):

1. Happy path: two Behaviors, the model wrong once on the second. The node trace equals the expected sequence exactly, the run ends at `finish`, and after `yes` the project contains both tests and the implementation.
2. A test that passes at Red and also passes against the HEAD implementation routes back to `write_test`; three in a row interrupt at `stuck`.
2b. A test that passes at Red but fails against the HEAD implementation marks the Behavior Satisfied and moves to `next_behavior`, and the Workspace files are unchanged afterwards.
3. A test file with a `SyntaxError` routes to `write_test`, never `write_impl`.
4. Green failing `max_attempts` times interrupts at `stuck`; a hint resumes `write_impl`, and the next prompt contains the hint.
5. A failing Probe interrupts at `clarify` before any file is written (`changes()` is empty).
6. A Sign-off answer other than `yes` or `abort` adds a Behavior and the run returns to `write_test`.
7. A Polish edit that breaks a test is reverted; files equal their pre-Polish contents.
8. A `FileEdit` with `../outside.py` is not written and counts as an attempt.
9. Answering `no` at `finish` leaves the project tree unchanged and the Workspace on disk.

### 13.3 CLI and config

- `harness-workflow build --plan <path> [--project .] [--model <id>]` in `workflow/cli.py`, added to `[project.scripts]`. It preflights the model the way `evals/runner.py` does, creates the Workspace, runs `sync_deps`, streams node updates (`stream_mode="updates"`) as one line each, and answers interrupts from stdin. `--no-hints` behaves as in M11: a Stuck Behavior is marked failed and the run moves on, `clarify` waives, and Sign-off answers yes. `finish` still asks for Approval. Each run writes the M11 summary JSON, so `examples/graphs/report.py` covers both.
- `[workflow]` in `harness.toml` and `config.py`: `model` (default: the Profile's model), `num_ctx` (default 16384), `max_attempts`, `max_polish_rounds`, `test_command`, `lint_commands`, `workspaces_dir`.
- Dependency: `uv add langgraph`. When `mypy --strict` rejects LangGraph types, wrap them in a typed helper in `workflow/`; list any remaining `type: ignore` in the status notes.

**Live check** (`tests/test_live_workflow.py`, `HARNESS_LIVE=1`): one kata from `~/projects/braid-practice/` with its Claude `plan.json`, `gemma4:12b`, hints off. The test passes when the run ends at `finish` or `stuck` without an exception; model quality is measured, not gated. Then rerun the M11 baseline (all five katas, `gemma4:e4b` and `gemma4:12b`, plus `llama3.1:8b` once it is pulled) through `harness-workflow build`, and put the new table next to M11's in the status notes. A drop from the spike is a bug in the port, not in the model.

**M13 validation:** given a Plan and a git project, `harness-workflow build` reaches an applied diff or a clean stop, with every model call pinned through the guard client and every command run in the Sandbox; all ten graph gates pass with a scripted model.

**Try it:** build a CLI from rung 2 of the practice ladder from a Claude Plan, apply the diff, and use the tool. Then add a feature to an existing hobby repo (rung 3).

---

## M14 — Plan loop (CLI)

The Plan loop from `examples/graphs/plan_graph.py`, with grounded research. Depends on 13.1. ADR 0004.

### 14.1 Research without tool calling

The research node takes two model calls and a harness step in between:

1. The model returns `ResearchQueries` (up to 6 items, each `{kind: "read" | "grep" | "list", target: str, why: str}`).
2. The harness runs them in-process with `resolve_inside(project, …)`: `read` returns up to 300 lines, `grep` is a Python regex over `git ls-files` with up to 50 hits, `list` is a directory listing. A target outside the project is refused and noted.
3. The model returns `Findings` (`list[{claim, evidence: "path:line"}]`). The harness checks that each cited path exists and the line is in range. A finding with a bad citation is kept but marked `unverified` and becomes an open question.

On rounds after the first, the human's `research: …` text goes into step 1's prompt.

### 14.2 The graph: `workflow/plan_loop.py`

Nodes `research`, `draft`, `critique` (fanned out with `Send`, one per angle), `synthesize`, `human_review`, `finalize`, routed as in the example.

- `draft` returns a `Plan`. Every Assumption needs a `probe` or a verified `evidence`; otherwise the draft is retried once with the offending Assumptions named, then those go into `open_questions`.
- `[workflow] critique_angles`, default `["feasibility", "scope", "testability", "risk"]`. Each angle has its own system prompt in `workflow/prompts.py` and returns `{points: list[str]}` (up to 3). Use a reducer keyed by round, as in the example.
- `synthesize` returns `open_questions` (up to 5): questions only the human can answer.
- `human_review` routes `approve` to `finalize`, `research: …` to `research`, and any other text to `draft`. At `max_rounds` (default 3) it goes to `finalize` with `"stopped at round limit"` in `notes`.
- `finalize` saves the Plan to `~/.braid/plans/<id>.json` and prints a Markdown rendering of it.

CLI: `harness-workflow plan "<goal>" [--project .]` prints the saved path; `harness-workflow build --plan <that path>` consumes it.

Gates (`tests/workflow/test_plan_loop.py`, `ScriptedModel`, fixture repo):

1. One round makes exactly one call per angle, `synthesize` runs once, and the critiques in state are tagged with that round.
2. `research: …`, revise text and `approve` each take their route. Round `max_rounds` finalizes without asking again.
3. A query whose target is outside the project is refused and nothing outside is read.
4. A finding citing a missing file or an out-of-range line is marked `unverified` and appears in `open_questions`.
5. An Assumption with neither a probe nor verified evidence triggers exactly one draft retry.
6. The finalized Plan loads with `Plan.load` and runs through `build_execution_graph` up to its first interrupt with the scripted model.

**Live check:** plan the five M11 katas locally with `gemma4:12b`, approving each at the first review that has no open questions. Run each local Plan through `harness-workflow build --no-hints`, and put the results in the status notes next to the Claude-Plan results for the same model. This comparison measures what the Plan loop is worth.

**M14 validation:** from a one-line goal the CLI produces a Plan whose every Assumption has a Probe or a checked citation, after at least one human review, and that Plan feeds M13 unchanged.

**Try it:** add a fifth Critique angle of your own (for example "what would the user of this tool complain about") through config and its prompt, and see whether it changes a Plan. Then add a new node yourself: a `prior_art` step between `research` and `draft` that searches `~/projects/braid-practice/` for similar code.

---

## M15 — Workflows in the TUI

Runs both loops from the TUI, with resume. ADRs: 0003, 0005.

- **Commands:** `/plan <goal>`, `/build [plan-id]` (most recent Plan when omitted), and `/runs` (recent runs with their state). Add them to the command table in `app.py`.
- **Progress:** one line per node update in the chat pane, styled like status lines, not like model replies.
- **Interrupts:** an inline widget modelled on `ApprovalPrompt` that shows the payload and takes a free-text answer. `finish` uses the real Approval path, so it emits `ApprovalRequested` and `ApprovalResolved` and holds the Prompt queue. Every other interrupt holds the queue too.
- **Checkpointing:** `langgraph-checkpoint-sqlite` at `~/.braid/workflows.db`, with `thread_id` = run id. `/build --resume <run-id>` and `/runs` read it.
- **Ctrl-C:** cancels the running node's task, kills its Sandbox process group, and pauses the run at its last checkpoint. It never applies anything.
- **Models:** a `[workflow] model` other than the loaded model is preflighted, and the loaded model is unloaded for the run, as in evals. The status bar shows the workflow model while the run is active.
- **Hook events:** add `WorkflowNodeFinished(run_id, node, duration_s, attempt)` and `WorkflowInterrupted(run_id, kind)` to `hooks.py`. Telemetry turns each into a span or metric.

Gates (`tests/test_app_workflows.py`, Textual pilot, `ScriptedModel`, `LocalSandbox`):

1. `/build` runs to a `stuck` interrupt; a prompt submitted meanwhile stays queued until the widget is answered.
2. The `finish` Approval emits both approval hook events, and `n` leaves the project unchanged.
3. Ctrl-C during a node, then `/build --resume`, finishes the run without re-running completed nodes (count `ScriptedModel` calls).
4. Killing the app mid-run and restarting with `/build --resume <id>` continues from the checkpoint.
5. Every node update emits `WorkflowNodeFinished`.

**M15 validation:** both loops run from the TUI end to end, every human pause holds the Prompt queue, and a cancelled or killed run resumes from its last checkpoint without repeating model calls.

**Try it:** for a week, start hobby work with `/plan` or a Claude Plan and `/build`. Kill braid mid-run once and resume. Then take rung 4: build `/export` in braid with braid.
