"""The Execution loop (ADR 0004, 0005): check Assumptions, make each Behavior Red then Green, Sign-off, Polish, finish.

Models write, code checks: every pass/fail decision is a pytest or Probe run in the Sandbox, a path rule, an Invariant
check in the Sandbox (F3), or a human. A Dispute from the implementer never routes by itself: it only triggers the
Invariant check on the disputed test. Node names and routing follow ``examples/graphs/exec_graph.py``.
"""

from __future__ import annotations

import functools
import inspect
import operator
import re
import shlex
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from harness.config import WorkflowConfig
from harness.sandbox import RunResult, Sandbox, Workspace
from harness.tools.builtin import resolve_inside
from harness.workflow import invariants as inv
from harness.workflow import prompts
from harness.workflow.model import WorkflowModel, WorkflowModelError
from harness.workflow.plan import Plan
from harness.workflow.records import BehaviorStats, Recorder, RunLog, UsageMeter, tail

EXECUTION = "tdd-v2-invariants"  # the graph variant, recorded in every summary (F2): change it when the graph's shape changes


def execution_variant(settings: WorkflowConfig) -> str:
    """``EXECUTION`` plus the switches that change the graph's behavior, for the summary's provenance."""
    return EXECUTION + ("" if settings.check_tests_at_red else "/no-red-check") + ("" if settings.show_invariants else "/hidden-rules")


RESERVED = {".git", ".venv", ".tmp", "__pycache__", ".pytest_cache"}
PROBE_TIMEOUT_S = 30.0
LAST: Literal["__end__"] = "__end__"  # equals langgraph END, which is typed as plain str
YES = {"y", "yes", "ok", "looks good", "lgtm"}


class FileEdit(BaseModel):
    path: str  # relative to the Workspace
    content: str  # full new file contents


class Edits(BaseModel):
    edits: list[FileEdit] = Field(min_length=1, max_length=3)
    rationale: str  # one sentence, logged


class Dispute(BaseModel):
    test: str  # the disputed test function's name
    reason: str


class ImplEdits(BaseModel):
    """``write_impl``'s reply: edits, or a Dispute (F3), or both."""

    edits: list[FileEdit] = Field(default_factory=list, max_length=3)
    rationale: str
    dispute: Dispute | None = None


class ExecState(TypedDict, total=False):
    plan: dict[str, Any]  # Plan.model_dump(); Sign-off may append Behaviors, clarify may add notes
    idx: int  # the Behavior being worked on
    attempts: int  # implementation attempts since the last hint
    red_tries: int  # rejected tests for this Behavior since the last hint
    hint: str
    last_run: str
    refused: bool  # the last implementation edit was refused, so there is nothing new to run
    passed: bool
    red_verdict: Literal["red", "passed", "rejected"]
    satisfied: bool
    phase: Literal["test", "impl", "skipped"]
    waived: Annotated[list[str], operator.add]
    failed: list[str]
    snapshot: dict[str, bytes]  # Workspace files as this Behavior found them: what a skip reverts to
    polish_snapshot: dict[str, bytes | None]  # files Polish changed, as they were (None: did not exist)
    polish_round: int
    polish_feedback: str
    edited: list[str]
    test_paths: list[str]
    test_owner: dict[str, int]  # test file -> the Behavior whose write_test wrote it
    invariant_verdict: Literal["ok", "violated"]
    dispute: dict[str, Any] | None  # the last Dispute on this Behavior and how the check went: shown at Stuck
    dispute_raised: dict[str, str] | None  # the Dispute in the last write_impl reply, until check_dispute reads it
    dispute_note: str  # a Dispute not upheld, for the next write_impl prompt
    revisit: dict[str, int] | None  # {"idx": earlier Behavior whose test is being fixed, "back": the one to return to}
    status: str
    recorder: dict[str, Any]  # Recorder.dump() after each node: the per-Behavior stats and elapsed time survive a resume


def initial_state(plan: Plan) -> ExecState:
    return {"plan": plan.model_dump(), "idx": 0, "attempts": 0, "red_tries": 0, "edited": [], "snapshot": {}}


def is_yes(answer: str) -> bool:
    return answer.strip().lower() in YES


def is_abort(answer: str) -> bool:
    return answer.strip().lower().startswith("abort")


def classify_red(result: RunResult, test_paths: list[str]) -> tuple[Literal["red", "passed", "rejected"], str]:
    """Red is pytest exit 1, or exit 2 from an import error naming a module that is not the test file itself."""
    if result.timed_out:
        return "rejected", "the test run timed out"
    if result.exit_code == 0:
        return "passed", "the test passes"
    if result.exit_code == 1:
        return "red", ""
    if result.exit_code == 2 and "SyntaxError" not in result.output:
        stems = {Path(p).stem for p in test_paths} | {p.removesuffix(".py").replace("/", ".") for p in test_paths}
        named = re.findall(r"No module named '([^']+)'", result.output) + re.findall(r"cannot import name '[^']+' from '([^']+)'", result.output)
        if "ModuleNotFoundError" in result.output or "ImportError" in result.output:
            if not named or any(name not in stems for name in named):
                return "red", ""
            return "rejected", "the test file imports itself"
    return "rejected", f"pytest exit {result.exit_code}: not a failing test (collection or syntax error?)"


class _NodeFn(Protocol):
    """The shape LangGraph's typing wants for a node: a callable whose parameter is named ``state``."""

    async def __call__(self, state: ExecState) -> dict[str, Any]: ...


class _Loop:
    def __init__(
        self, model: WorkflowModel, sandbox: Sandbox, workspace: Workspace, settings: WorkflowConfig, *,
        hints: bool, recorder: Recorder, log: RunLog | None,
    ) -> None:
        self.model, self.sandbox, self.ws, self.cfg = model, sandbox, workspace, settings
        self.hints, self.rec, self.runlog = hints, recorder, log
        self.meter = UsageMeter(model)
        self.node_started = time.monotonic()
        venv_bin = workspace.path / ".venv" / "bin"
        self.env = {
            "PATH": f"{venv_bin}:/usr/bin:/bin:/usr/sbin:/sbin", "VIRTUAL_ENV": str(workspace.path / ".venv"),
            "PYTHONDONTWRITEBYTECODE": "1",  # no stale .pyc when a file is rewritten within the same second
        }

    # --- helpers -------------------------------------------------------------------------------------------
    def log(self, node: str, state: ExecState, **fields: Any) -> None:
        self.rec.trace.append(node)
        usage = self.rec.add_usage(self.meter.take(), state.get("idx"))  # the API calls this node made, if any
        if self.runlog:
            self.runlog.write(node, behavior=state.get("idx"), node_seconds=round(time.monotonic() - self.node_started, 2), **fields, **usage)

    async def sh(self, command: str, timeout: float | None = None) -> RunResult:
        return await self.sandbox.run(
            command, cwd=self.ws.path, policy=self.ws.policy(), timeout_s=timeout or self.cfg.command_timeout_s, env=self.env,
        )

    async def pytest(self, target: str = "") -> RunResult:
        return await self.sh(f"{self.cfg.test_command} {target}".strip())

    @staticmethod
    def ran(r: RunResult, **extra: Any) -> dict[str, Any]:
        return {"exit_code": r.exit_code, "timed_out": r.timed_out, "seconds": round(r.duration_s, 2), "output_tail": tail(r.output), **extra}

    def plan(self, state: ExecState) -> Plan:
        return Plan.model_validate(state["plan"])

    def current(self, state: ExecState) -> BehaviorStats:
        return self.rec.behaviors[state["idx"]]

    async def capture(self) -> dict[str, bytes]:
        return {rel: (self.ws.path / rel).read_bytes() for rel in await self.ws.list_files()}

    async def restore(self, files: Mapping[str, bytes]) -> None:
        for rel in await self.ws.list_files():
            if rel not in files:
                (self.ws.path / rel).unlink()
        for rel, data in files.items():
            self.put(rel, data)

    def put(self, rel: str, data: bytes | None) -> None:
        path = self.ws.path / rel
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    async def context(self, state: ExecState) -> str:
        plan = self.plan(state)
        files: dict[str, str] = {}
        for rel in dict.fromkeys([*plan.files, *state.get("edited", [])]):
            if (self.ws.path / rel).is_file():
                files[rel] = (self.ws.path / rel).read_text(errors="replace")
        return prompts.context(plan, state.get("idx", 0), await self.ws.list_files(), files, rules=self.cfg.show_invariants)

    async def ask[E: (Edits, ImplEdits)](self, prompt: str, schema: type[E]) -> tuple[E | None, float, str]:
        start = time.monotonic()
        edits: E | None = None
        error = ""
        try:
            edits = await self.model.complete(prompts.SYSTEM, prompt, schema)
        except WorkflowModelError as exc:
            error = str(exc)
        elapsed = time.monotonic() - start
        self.rec.model_calls += 1
        self.rec.model_seconds += elapsed
        return edits, elapsed, error

    def apply(
        self, edits: Edits | ImplEdits | None, allow: Literal["tests", "impl"], before: dict[str, bytes | None] | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        """Write the edits that pass the path rules; return (written, refusals). ``before`` collects what each path held."""
        written: dict[str, str] = {}
        refused: list[str] = []
        if edits is None:
            return written, ["the model returned no valid edits"]
        for edit in edits.edits:
            try:
                path = resolve_inside(self.ws.path, edit.path)
                rel = str(path.relative_to(self.ws.path))
                if RESERVED & set(Path(rel).parts) or path == self.ws.path:
                    raise ValueError(f"{edit.path!r} is reserved")
                if (allow == "tests") != rel.startswith("tests/"):
                    raise ValueError(
                        f"{'only files under tests/' if allow == 'tests' else 'not files under tests/'} may be edited now: {edit.path}"
                    )
                if before is not None and rel not in before:
                    before[rel] = path.read_bytes() if path.is_file() else None
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(edit.content)
                written[rel] = edit.content
            except ValueError as exc:
                refused.append(str(exc))
        return written, refused

    # --- Invariants (F3) -----------------------------------------------------------------------------------
    def python(self) -> str:
        """The interpreter the checker runs on: the test command's, when it names one."""
        first = shlex.split(self.cfg.test_command)[0] if self.cfg.test_command.strip() else ""
        return first if "python" in Path(first).name else "python3"

    async def check(self, plan: Plan, paths: list[str], test: str | None = None) -> inv.CheckResult:
        """Evaluate the Plan's Invariants on the examples asserted in ``paths`` (only ``test``'s, when given), in the
        Sandbox. Model-written test files and Invariants never run in the harness itself."""
        examples: list[inv.Example] = []
        skipped: list[str] = []
        for rel in paths:
            file = self.ws.path / rel
            if not file.is_file():
                continue
            found = inv.extract(file.read_text(errors="replace"), rel)
            examples += found.examples
            skipped += found.skipped
        if test:
            name = test.split("::")[-1].split("[")[0].strip()
            examples = [e for e in examples if e.test.split("::")[-1].split("[")[0] == name]
            skipped = [s for s in skipped if f"::{name}" in s]
        if not examples or not plan.invariants:
            return inv.CheckResult(examples=len(examples), skipped=skipped)
        scratch = self.ws.path / ".tmp" / "invariants"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "check.py").write_text(inv.CHECKER)
        (scratch / "input.json").write_text(inv.checker_input(examples, plan.invariants))
        r = await self.sh(f"{shlex.quote(self.python())} -I .tmp/invariants/check.py .tmp/invariants/input.json", PROBE_TIMEOUT_S)
        if r.timed_out:
            return inv.CheckResult(examples=len(examples), skipped=skipped, error="the invariant check timed out")
        return inv.read_checker_output(r.output, len(examples), skipped)

    def test_files(self, state: ExecState) -> list[str]:
        return sorted(state.get("test_owner", {})) or sorted(
            str(p.relative_to(self.ws.path)) for p in (self.ws.path / "tests").rglob("test*.py")
        )

    # --- stage 1: assumption check -------------------------------------------------------------------------
    async def check_assumptions(self, state: ExecState) -> dict[str, Any]:
        failed: list[str] = []
        results: list[dict[str, Any]] = []
        for a in self.plan(state).assumptions:
            if a.text in state.get("waived", []):
                results.append({"text": a.text, "result": "waived"})
            elif not a.probe:
                results.append({"text": a.text, "result": "no probe"})
            else:
                r = await self.sh(a.probe, PROBE_TIMEOUT_S)
                results.append({"text": a.text, "probe": a.probe, "result": "ok" if r.exit_code == 0 else "failed", **self.ran(r)})
                if r.exit_code != 0:
                    failed.append(a.text)
        self.log("check_assumptions", state, probes=results, failed=failed)
        return {"failed": failed}

    def after_check(self, state: ExecState) -> Literal["clarify", "write_test"]:
        return "clarify" if state["failed"] or self.plan(state).open_questions else "write_test"

    def clarify(self, state: ExecState) -> dict[str, Any]:
        plan = self.plan(state)
        if self.hints:
            answer = str(interrupt({"kind": "clarify", "failed": state["failed"], "open_questions": plan.open_questions}))
        else:
            answer = "waived (no-hints run)"
        self.log("clarify", state, failed=state["failed"], open_questions=plan.open_questions, answer=answer)
        if is_abort(answer):
            return {"status": "aborted"}
        plan = plan.model_copy(update={"open_questions": [], "notes": [*plan.notes, answer]})
        return {"waived": state["failed"], "plan": plan.model_dump()}

    @staticmethod
    def after_clarify(state: ExecState) -> Literal["check_assumptions", "__end__"]:
        return LAST if state.get("status") == "aborted" else "check_assumptions"

    # --- stage 2: make it work -----------------------------------------------------------------------------
    async def write_test(self, state: ExecState) -> dict[str, Any]:
        idx = state.get("idx", 0)
        plan = self.plan(state)
        if idx >= len(self.rec.behaviors):
            self.rec.behaviors.append(BehaviorStats(plan.behaviors[idx]))
        fresh = state.get("phase") != "test"  # not a rewrite of a rejected test
        out: dict[str, Any] = {"idx": idx, "phase": "test"}
        if fresh:
            self.rec.behavior_started = time.monotonic()
            out.update(snapshot=await self.capture(), red_tries=0, satisfied=False, last_run="", dispute=None, dispute_note="")
        elif not state.get("revisit"):  # a rewrite starts from the Workspace as the Behavior found it
            await self.restore(state["snapshot"])  # so a rejected test file cannot linger
        prompt = prompts.write_test(await self.context(cast(ExecState, {**state, **out})), idx, plan.behaviors[idx], state.get("hint", ""), "" if fresh else state.get("last_run", ""))
        edits, elapsed, error = await self.ask(prompt, Edits)
        written, refused = self.apply(edits, "tests")
        out["test_paths"] = [*written]
        out["test_owner"] = {**state.get("test_owner", {}), **{rel: idx for rel in written}}
        out["edited"] = [*dict.fromkeys([*state.get("edited", []), *written])]
        if refused:
            out["last_run"] = "Edit refused: " + "; ".join(refused)
        self.log(
            "write_test", state, attempt=state.get("red_tries", 0) + 1 if not fresh else 1, prompt=prompt,
            reply=edits.model_dump() if edits else None, model_error=error, written=[*written], refused=refused,
            model_seconds=round(elapsed, 2),
        )
        return out

    async def run_red(self, state: ExecState) -> dict[str, Any]:
        tries = state.get("red_tries", 0)
        paths = state.get("test_paths", [])
        if not paths:
            self.current(state).red_rejections += 1
            self.log("run_red", state, attempt=tries + 1, verdict="rejected", reason="no test file was written", exit_code=None)
            return {"red_verdict": "rejected", "red_tries": tries + 1}
        r = await self.pytest(" ".join(paths))
        verdict, reason = classify_red(r, paths)
        self.log("run_red", state, attempt=tries + 1, verdict=verdict, reason=reason, **self.ran(r))
        if verdict == "red":
            return {"red_verdict": "red", "last_run": tail(r.output)}
        if verdict == "passed":
            return {"red_verdict": "passed", "last_run": tail(r.output)}
        self.current(state).red_rejections += 1
        return {"red_verdict": "rejected", "red_tries": tries + 1, "last_run": f"{reason}\n{tail(r.output)}"}

    def after_red(self, state: ExecState) -> Literal["write_test", "write_impl", "check_satisfied", "check_invariants", "stuck"]:
        if state.get("revisit") and state["red_verdict"] != "rejected":  # a corrected earlier test: check it, then go back
            return "check_invariants"
        if state["red_verdict"] == "red":
            return "check_invariants" if self.plan(state).invariants and self.cfg.check_tests_at_red else "write_impl"
        if state["red_verdict"] == "passed":
            return "check_satisfied"
        return "stuck" if state["red_tries"] >= self.cfg.max_attempts else "write_test"

    async def check_invariants(self, state: ExecState) -> dict[str, Any]:
        """After Red, before implementation sees the test: every expected value it asserts must satisfy the Plan's
        Invariants. A violation is a Red rejection, sent back to ``write_test`` with the rule and the example named."""
        result = await self.check(self.plan(state), state.get("test_paths", []))
        self.log("check_invariants", state, **result.summary())
        if not result.violations:
            return {"invariant_verdict": "ok"}
        self.current(state).red_rejections += 1
        return {"invariant_verdict": "violated", "red_tries": state.get("red_tries", 0) + 1, "last_run": inv.feedback(result.violations)}

    def after_invariants(self, state: ExecState) -> Literal["write_impl", "write_test", "resume_behavior", "stuck"]:
        if state["invariant_verdict"] == "violated":
            return "stuck" if state["red_tries"] >= self.cfg.max_attempts else "write_test"
        return "resume_behavior" if state.get("revisit") else "write_impl"

    def resume_behavior(self, state: ExecState) -> dict[str, Any]:
        """An earlier Behavior's test was corrected after an upheld Dispute: back to the Behavior that disputed it, whose
        suite now runs with the corrected test."""
        revisit = state["revisit"] or {}
        self.log("resume_behavior", state, corrected=state.get("test_paths", []), back_to=revisit.get("back"))
        return {"idx": revisit["back"], "revisit": None, "phase": "impl", "refused": False, "red_tries": 0}

    async def check_satisfied(self, state: ExecState) -> dict[str, Any]:
        """The new test passed at Red. Run it against the implementation as it was at HEAD: if it fails there, earlier
        code in this run already covers the Behavior (satisfied); if it passes there too, it checks nothing (vacuous)."""
        paths = state.get("test_paths", [])
        swapped = [c for c in await self.ws.changes() if not c.path.startswith("tests/")]
        current = {c.path: (self.ws.path / c.path).read_bytes() if (self.ws.path / c.path).is_file() else None for c in swapped}
        head = {c.path: await self.ws.read_head(c.path) for c in swapped}
        try:
            for rel, data in head.items():
                self.put(rel, data)
            r = await self.pytest(" ".join(paths))
        finally:  # the Workspace always gets its current files back
            for rel, data in current.items():
                self.put(rel, data)
        satisfied = r.exit_code != 0 or r.timed_out
        self.log("check_satisfied", state, satisfied=satisfied, swapped=[*head], **self.ran(r))
        if satisfied:
            stats = self.current(state)
            stats.satisfied = True
            return {"satisfied": True}
        self.current(state).red_rejections += 1
        return {"satisfied": False, "red_tries": state.get("red_tries", 0) + 1, "last_run": prompts.vacuous_feedback(tail(r.output))}

    def after_satisfied(self, state: ExecState) -> Literal["next_behavior", "write_test", "stuck"]:
        if state["satisfied"]:
            return "next_behavior"
        return "stuck" if state["red_tries"] >= self.cfg.max_attempts else "write_test"

    async def write_impl(self, state: ExecState) -> dict[str, Any]:
        stats = self.current(state)
        stats.attempts += 1
        attempts = state.get("attempts", 0) + 1
        behavior = self.plan(state).behaviors[state["idx"]]
        prompt = prompts.write_impl(
            await self.context(state), behavior, state.get("hint", ""), state.get("last_run", ""), state.get("dispute_note", ""),
        )
        edits, elapsed, error = await self.ask(prompt, ImplEdits)
        written, refused = self.apply(edits, "impl")
        dispute = edits.dispute if edits is not None else None
        if edits is not None and not written and not refused and dispute is None:
            refused = ["the reply changed no file and raised no dispute"]
        out: dict[str, Any] = {"attempts": attempts, "phase": "impl", "edited": [*dict.fromkeys([*state.get("edited", []), *written])]}
        out["refused"] = bool(refused)
        out["dispute_raised"] = dispute.model_dump() if dispute else None
        if refused:
            out["last_run"] = "Edit refused: " + "; ".join(refused)
        self.log(
            "write_impl", state, attempt=attempts, prompt=prompt, reply=edits.model_dump() if edits else None,
            model_error=error, written=[*written], refused=refused, model_seconds=round(elapsed, 2),
            dispute=out["dispute_raised"],
        )
        return out

    def after_impl(self, state: ExecState) -> Literal["check_dispute", "run_green"]:
        return "check_dispute" if state.get("dispute_raised") else "run_green"

    async def check_dispute(self, state: ExecState) -> dict[str, Any]:
        """The implementer disputed a test. Code decides (ADR 0004): the Plan's Invariants on that test's examples.

        A violation is upheld: the test goes back to the ``write_test`` of the Behavior that wrote it, with the reason and
        the rule, and this implementation attempt is not counted. Otherwise the test stands, the attempt counts, and the
        reason goes into the next prompt."""
        raised = state.get("dispute_raised") or {}
        test, reason = raised.get("test", ""), raised.get("reason", "")
        result = await self.check(self.plan(state), self.test_files(state), test)
        stats = self.current(state)
        stats.disputes += 1
        record: dict[str, Any] = {"test": test, "reason": reason, "upheld": bool(result.violations), **result.summary()}
        self.log("check_dispute", state, **record)
        out: dict[str, Any] = {"dispute": record, "dispute_raised": None}
        if not result.violations:
            why = ("no Invariant in the Plan" if not self.plan(state).invariants
                   else "no literal assert found in that test" if not result.examples
                   else f"all {result.checked} checks passed" + (f", {len(result.unchecked)} unchecked" if result.unchecked else ""))
            out["dispute_note"] = (f"Your dispute of {test} was checked against the Plan's rules and not upheld ({why}). "
                                   f"The test stands; make it pass. Your reason was: {reason}")
            out["refused"] = False  # run the suite even when the reply wrote nothing, so the next prompt has its output
            return out
        stats.disputes_upheld += 1
        stats.attempts -= 1  # an upheld Dispute does not use up an implementation attempt
        owner = state.get("test_owner", {}).get(result.violations[0].test.split("::")[0], state["idx"])
        out.update(
            attempts=state["attempts"] - 1, phase="test", red_tries=0,
            last_run=f"The implementer disputed {test}: {reason}\n{inv.feedback(result.violations)}",
        )
        if owner < state["idx"]:  # an earlier Behavior's test: fix it there, then come back
            out.update(idx=owner, revisit={"idx": owner, "back": state["idx"]})
        return out

    @staticmethod
    def after_dispute(state: ExecState) -> Literal["write_test", "run_green"]:
        return "write_test" if state.get("phase") == "test" else "run_green"

    async def run_green(self, state: ExecState) -> dict[str, Any]:
        if state.get("refused"):  # nothing was written, so there is nothing to run
            self.log("run_green", state, attempt=state["attempts"], skipped="edit refused", exit_code=None)
            return {"passed": False}
        r = await self.pytest()  # the whole suite, so earlier Behaviors cannot regress
        self.log("run_green", state, attempt=state["attempts"], **self.ran(r))
        return {"passed": r.exit_code == 0 and not r.timed_out, "last_run": tail(r.output)}

    def after_green(self, state: ExecState) -> Literal["next_behavior", "write_impl", "stuck"]:
        if state["passed"]:
            return "next_behavior"
        return "write_impl" if state["attempts"] < self.cfg.max_attempts else "stuck"

    async def stuck(self, state: ExecState) -> dict[str, Any]:
        behavior = self.plan(state).behaviors[state["idx"]]
        dispute = state.get("dispute")
        if self.hints:
            payload: dict[str, Any] = {"kind": "stuck", "behavior": behavior, "last_run": state.get("last_run", "")}
            if dispute:
                payload["dispute"] = dispute
            answer = str(interrupt(payload))
        else:
            answer = "skip"
        self.rec.stuck += 1
        revisit = state.get("revisit")
        idx = revisit["back"] if revisit else state["idx"]  # stuck fixing an earlier test: the disputing Behavior is stuck
        stats = self.rec.behaviors[idx]
        self.log("stuck", state, answer=answer, last_run=tail(state.get("last_run", "")), dispute=dispute)
        if is_abort(answer):
            return {"status": "aborted"}
        if answer.strip().lower().startswith("skip"):  # mark failed and put the Workspace back as this Behavior found it
            stats.green, stats.note = False, "stuck" + (" after a dispute" if dispute else "")
            stats.seconds = time.monotonic() - self.rec.behavior_started
            await self.restore(state["snapshot"])
            return {"phase": "skipped", "idx": idx + 1, "attempts": 0, "red_tries": 0, "hint": "", "last_run": "",
                    "revisit": None, "dispute": None, "dispute_note": ""}
        stats.hints += 1
        return {"attempts": 0, "red_tries": 0, "hint": answer}

    def after_stuck(self, state: ExecState) -> Literal["write_test", "write_impl", "sign_off", "__end__"]:
        if state.get("status") == "aborted":
            return LAST
        if state.get("phase") == "skipped":
            return "write_test" if state["idx"] < len(self.plan(state).behaviors) else "sign_off"
        return "write_test" if state.get("phase") == "test" else "write_impl"

    def next_behavior(self, state: ExecState) -> dict[str, Any]:
        stats = self.current(state)
        stats.green = True
        stats.seconds = time.monotonic() - self.rec.behavior_started
        self.log("next_behavior", state, satisfied=stats.satisfied, seconds=round(stats.seconds, 2))
        return {"idx": state["idx"] + 1, "attempts": 0, "phase": "impl", "hint": "", "last_run": "", "dispute": None, "dispute_note": ""}

    def after_next(self, state: ExecState) -> Literal["write_test", "sign_off"]:
        return "write_test" if state["idx"] < len(self.plan(state).behaviors) else "sign_off"

    # --- stage 3: make it right ----------------------------------------------------------------------------
    async def sign_off(self, state: ExecState) -> dict[str, Any]:
        plan = self.plan(state)
        demo = await self.sh(plan.demo, PROBE_TIMEOUT_S) if plan.demo else None
        suite = await self.pytest("-rA")
        passing = [line.removeprefix("PASSED ") for line in suite.output.splitlines() if line.startswith("PASSED ")]
        answer = str(interrupt({"kind": "sign_off", "demo": demo.output if demo else "", "passing": passing})) if self.hints else "yes"
        self.log(
            "sign_off", state, answer=answer, passing=passing,
            demo=self.ran(demo) if demo else None, demo_command=plan.demo,
        )
        if is_abort(answer):
            return {"status": "aborted"}
        if is_yes(answer):
            return {}
        # Anything else becomes a new Behavior, so the work loop runs again.
        plan = plan.model_copy(update={"behaviors": [*plan.behaviors, answer]})
        return {"plan": plan.model_dump(), "phase": "impl"}

    def after_sign_off(self, state: ExecState) -> Literal["polish", "write_test", "finish", "__end__"]:
        if state.get("status") == "aborted":
            return LAST
        if state["idx"] < len(self.plan(state).behaviors):
            return "write_test"
        return "polish" if self.cfg.max_polish_rounds > 0 else "finish"

    async def polish(self, state: ExecState) -> dict[str, Any]:
        round_ = state.get("polish_round", 0) + 1
        prompt = prompts.polish(await self.context(state), state.get("polish_feedback", ""))
        edits, elapsed, error = await self.ask(prompt, Edits)
        before: dict[str, bytes | None] = {}
        written, refused = self.apply(edits, "impl", before)
        self.log(
            "polish", state, attempt=round_, prompt=prompt, reply=edits.model_dump() if edits else None, model_error=error,
            written=[*written], refused=refused, model_seconds=round(elapsed, 2),
        )
        return {"polish_snapshot": before, "polish_round": round_}

    async def verify_polish(self, state: ExecState) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        failure = ""
        for command in [self.cfg.test_command, *self.cfg.lint_commands]:
            r = await self.sh(command)
            checks.append({"command": command, **self.ran(r)})
            if r.exit_code != 0 or r.timed_out:
                failure = f"$ {command}\n{tail(r.output)}"
                break
        if not failure:
            self.rec.polish = "kept"
            self.log("verify_polish", state, kept=True, checks=checks)
            return {"polish_feedback": ""}
        for rel, data in state.get("polish_snapshot", {}).items():
            self.put(rel, data)
        self.rec.polish = "reverted"
        self.log("verify_polish", state, kept=False, checks=checks)
        return {"polish_feedback": failure}

    def after_verify(self, state: ExecState) -> Literal["polish", "finish"]:
        if state.get("polish_feedback") and state["polish_round"] < self.cfg.max_polish_rounds:
            return "polish"
        return "finish"

    async def finish(self, state: ExecState) -> dict[str, Any]:
        changes = await self.ws.changes()
        payload = {
            "kind": "finish", "workspace": str(self.ws.path),
            "changes": [{"path": c.path, "status": c.status, "flags": sorted(c.flags)} for c in changes],
            "diff": await self.ws.diff(),
        }
        answer = str(interrupt(payload))
        status, detail = "kept", ""
        if is_yes(answer):
            try:
                await self.ws.apply()
                await self.ws.remove()
                status = "applied"
            except ValueError as exc:  # flagged or conflicting changes: the user keeps the Workspace
                status, detail = "apply_refused", str(exc)
        self.log("finish", state, answer=answer, outcome=status, detail=detail, changes=payload["changes"])
        return {"status": status}

    # --- Spend cap -----------------------------------------------------------------------------------------
    def spend_cap(self, state: ExecState) -> dict[str, Any]:
        """Terminal node: the run cost more than ``max_cost_usd``. Nothing was applied and nothing is asked."""
        self.log("spend_cap", state, spent=round(self.rec.cost_usd, 6), cap=self.cfg.max_cost_usd)
        return {"status": "spend_cap"}

    def gate(self, route: Callable[[ExecState], str]) -> Callable[[ExecState], str]:
        """``route``, unless the Spend cap has been passed: every node boundary then goes to ``spend_cap`` instead."""
        @functools.wraps(route)
        def gated(state: ExecState) -> str:
            return "spend_cap" if self.rec.cost_usd > self.cfg.max_cost_usd else route(state)

        return gated

    def timed(self, node: Callable[[ExecState], Any]) -> _NodeFn:
        """Stamp the node's start, so its log line carries the node's wall time (``node_seconds``)."""
        async def run(state: ExecState) -> dict[str, Any]:
            self.node_started = time.monotonic()
            result = node(state)
            update = cast(dict[str, Any], await result if inspect.isawaitable(result) else result)
            return {**update, "recorder": self.rec.dump()}

        return run

    def graph(self) -> StateGraph[ExecState]:
        g = StateGraph(ExecState)
        for name in (
            "check_assumptions", "clarify", "write_test", "run_red", "check_satisfied", "check_invariants", "resume_behavior",
            "write_impl", "check_dispute", "run_green", "stuck", "next_behavior", "sign_off", "polish", "verify_polish",
            "finish", "spend_cap",
        ):
            g.add_node(name, self.timed(getattr(self, name)))
        gate = self.gate

        def to(name: str) -> Callable[[ExecState], str]:  # a fixed edge that still honors the Spend cap
            return gate(lambda _state: name)

        def edges(source: str, route: Callable[[ExecState], str], targets: list[str]) -> None:
            g.add_conditional_edges(source, gate(route), [*targets, "spend_cap"])

        g.add_edge(START, "check_assumptions")
        edges("check_assumptions", self.after_check, ["clarify", "write_test"])
        edges("clarify", self.after_clarify, ["check_assumptions", END])
        edges("write_test", to("run_red"), ["run_red"])
        edges("run_red", self.after_red, ["write_test", "write_impl", "check_satisfied", "check_invariants", "stuck"])
        edges("check_invariants", self.after_invariants, ["write_impl", "write_test", "resume_behavior", "stuck"])
        edges("resume_behavior", to("run_green"), ["run_green"])
        edges("check_satisfied", self.after_satisfied, ["next_behavior", "write_test", "stuck"])
        edges("write_impl", self.after_impl, ["check_dispute", "run_green"])
        edges("check_dispute", self.after_dispute, ["write_test", "run_green"])
        edges("run_green", self.after_green, ["next_behavior", "write_impl", "stuck"])
        edges("stuck", self.after_stuck, ["write_test", "write_impl", "sign_off", END])
        edges("next_behavior", self.after_next, ["write_test", "sign_off"])
        edges("sign_off", self.after_sign_off, ["polish", "write_test", "finish", END])
        edges("polish", to("verify_polish"), ["verify_polish"])
        edges("verify_polish", self.after_verify, ["polish", "finish"])
        g.add_edge("spend_cap", END)
        g.add_edge("finish", END)
        return g


def build_execution_graph(
    model: WorkflowModel, sandbox: Sandbox, workspace: Workspace, settings: WorkflowConfig, *,
    hints: bool = True, recorder: Recorder | None = None, log: RunLog | None = None,
) -> StateGraph[ExecState]:
    """The Execution loop as a LangGraph ``StateGraph``. ``hints=False`` is the baseline mode: Stuck skips the
    Behavior, ``clarify`` waives and Sign-off says yes (``finish`` still asks). ``recorder`` and ``log`` collect results."""
    return _Loop(model, sandbox, workspace, settings, hints=hints, recorder=recorder or Recorder(), log=log).graph()


Ask = Callable[[dict[str, Any]], str | Awaitable[str]]


async def drive_graph[S](
    graph: StateGraph[Any], state: S, ask: Ask, *,
    on_update: Callable[[str, dict[str, Any]], None] | None = None, on_start: Callable[[str], None] | None = None,
    thread_id: str = "exec", checkpointer: BaseCheckpointSaver[Any] | None = None, resume: bool = False,
    on_resume: Callable[[dict[str, Any]], None] | None = None,
) -> S:
    """Run ``graph`` (either loop) to the end, answering each interrupt with ``ask(payload)``; returns the final state.

    ``checkpointer`` keeps the run's state per ``thread_id`` (default: in memory). ``resume=True`` ignores ``state`` and
    continues the thread from its last checkpoint: a node that was cut off runs again, a pending interrupt is asked
    again, and nothing that finished is repeated. ``on_resume`` gets the checkpointed values first."""
    compiled = graph.compile(checkpointer=checkpointer or InMemorySaver())
    config: Any = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
    run_input: Any = state
    if resume:
        snapshot = await compiled.aget_state(config)
        if not snapshot.values:
            raise ValueError(f"no checkpoint for run {thread_id}")
        if on_resume:
            on_resume(snapshot.values)
        if not snapshot.next:
            return cast(S, snapshot.values)
        run_input = None
    while True:
        pending: dict[str, Any] | None = None
        events = cast(AsyncIterator[tuple[str, Any]], compiled.astream(run_input, config, stream_mode=["updates", "tasks"]))
        async for mode, data in events:
            if mode == "tasks":
                if on_start and "input" in data:  # a task starting (its result event has no "input")
                    on_start(data["name"])
                continue
            for node, value in data.items():
                if node == "__interrupt__":
                    pending = value[0].value
                elif on_update:
                    on_update(node, value or {})
        if pending is None:
            return cast(S, (await compiled.aget_state(config)).values)
        answer = ask(pending)
        run_input = Command(resume=await answer if inspect.isawaitable(answer) else answer)
