"""Execution loop on a real model, under Seatbelt (M11.4): exec_graph.py with the scripted parts replaced.

    uv run --with langgraph python examples/graphs/exec_live.py ~/projects/braid-practice/roman-numerals \
        --model gemma4:e4b --no-hints

Reads <kata>/plan.json (made by claude_plan.sh). The kata is copied to a temp Workspace without .git and .venv,
`uv sync --frozen` runs there outside the Sandbox, and every pytest run, Probe and demo runs under a Seatbelt
profile: writes only in the Workspace, no network, no reads under $HOME except the Python toolchain.
Each run writes <kata>/runs/<timestamp>-<model>.json; report.py turns those into a table.

Throwaway code: it is held to "runs and reports honestly", not to mypy --strict. M12/M13 rebuild it properly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, run_preflight, system_gpu_limit

MAX_ATTEMPTS = 3
MAX_FILE_LINES = 300
MAX_RUN_LINES = 40
TIMEOUT_S = 120.0
MAX_OUTPUT = 64 * 1024
SKIP_DIRS = {".venv", ".git", ".tmp", "__pycache__", ".pytest_cache", "runs"}


# --- Model: structured replies through the guard client ------------------------------------------------------

class FileEdit(BaseModel):
    path: str  # relative to the Workspace
    content: str  # full new file contents


class Edits(BaseModel):
    edits: list[FileEdit]  # 1-3 files
    rationale: str  # one sentence, logged


class LiveModel:
    """Every request goes through OllamaClient.chat(format=schema), so num_ctx stays pinned."""

    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client, self.model = client, model
        self.calls = 0
        self.model_seconds = 0.0

    async def edits(self, system: str, prompt: str) -> Edits:
        error = ""
        for _ in range(2):  # one retry with the validation error appended
            start = time.monotonic()
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt + error}]
            text = "".join([c.text async for c in self.client.chat(
                self.model, messages, format=Edits.model_json_schema(), options={"temperature": 0.2})])
            self.calls += 1
            self.model_seconds += time.monotonic() - start
            try:
                return Edits.model_validate_json(text)
            except ValidationError as exc:
                error = f"\n\nYour last reply was invalid ({exc.error_count()} errors: {str(exc)[:300]}). Reply with valid JSON only."
        return Edits(edits=[], rationale="model returned invalid JSON twice")


# --- Sandbox: sandbox-exec with a generated profile (spec 12.1) ---------------------------------------------

def toolchain_paths(workdir: Path) -> list[Path]:
    python = (workdir / ".venv" / "bin" / "python").resolve()
    return [workdir / ".venv", python.parent.parent]  # the venv, and the interpreter's prefix (uv keeps it under ~)


def sbpl(workdir: Path, readable: list[Path]) -> str:
    def q(p: Path) -> str:
        return '"' + str(p.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'

    home = Path.home()
    lines = [
        "(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)",
        f"(allow file-write* (subpath {q(workdir)}) (literal \"/dev/null\") (literal \"/dev/tty\"))",
        f"(deny file-read* (subpath {q(home)}))",
        f"(allow file-read* {' '.join(f'(subpath {q(p)})' for p in [workdir, *readable])} (literal {q(home)}))",
        "(allow file-read-metadata)",  # without this Python cannot start
    ]
    return "\n".join(lines)


@dataclass
class RunResult:
    code: int
    output: str
    timed_out: bool = False


class Seatbelt:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir.resolve()
        (self.workdir / ".tmp").mkdir(exist_ok=True)
        self.profile = self.workdir.parent / f"{self.workdir.name}.sb"  # outside the Workspace: the model cannot edit it
        self.profile.write_text(sbpl(self.workdir, toolchain_paths(self.workdir)))

    async def run(self, command: str, timeout_s: float = TIMEOUT_S) -> RunResult:
        env = {
            "PATH": f"{self.workdir / '.venv' / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(self.workdir), "LANG": "en_US.UTF-8", "TERM": "dumb", "TMPDIR": str(self.workdir / ".tmp"),
            "VIRTUAL_ENV": str(self.workdir / ".venv"),
        }
        proc = await asyncio.create_subprocess_exec(
            "sandbox-exec", "-f", str(self.profile), "/bin/sh", "-c", command,
            cwd=self.workdir, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.5)
            out, _ = await proc.communicate()
        text = out.decode(errors="replace").strip()
        if len(text) > MAX_OUTPUT:
            text = f"[… trimmed {len(text) - MAX_OUTPUT} bytes …]\n" + text[-MAX_OUTPUT:]
        return RunResult(proc.returncode if proc.returncode is not None else -1, text, timed_out)


# --- Workspace ------------------------------------------------------------------------------------------------

def make_workspace(kata: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix=f"exec-live-{kata.name}-")).resolve() / "ws"
    shutil.copytree(kata, work, ignore=shutil.ignore_patterns(".git", ".venv", "plan.json", "runs", "__pycache__", ".pytest_cache"))
    subprocess.run(["uv", "sync", "--frozen"], cwd=work, check=True, capture_output=True)  # outside the Sandbox
    return work


def resolve_inside(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if rel.startswith("/") or not path.is_relative_to(root.resolve()) or SKIP_DIRS & set(path.relative_to(root.resolve()).parts):
        raise ValueError(f"path escapes the Workspace or is reserved: {rel}")
    return path


def file_tree(root: Path) -> str:
    files = sorted(
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and not SKIP_DIRS & set(p.relative_to(root).parts) and not p.name.endswith(".lock")
    )
    return "\n".join(files[:200])


def trimmed(text: str, limit: int) -> str:
    lines = text.splitlines()
    return text if len(lines) <= limit else "\n".join(lines[:limit]) + f"\n[… {len(lines) - limit} more lines …]"


def tail(text: str, limit: int = MAX_RUN_LINES) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


# --- Metrics ------------------------------------------------------------------------------------------------

@dataclass
class BehaviorStats:
    behavior: str
    green: bool = False
    attempts: int = 0
    red_rejections: int = 0
    hints: int = 0
    seconds: float = 0.0
    satisfied: bool = False  # the new test passed at Red: earlier code already does this, so no implementation attempt
    note: str = ""


@dataclass
class Recorder:
    behaviors: list[BehaviorStats] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    stuck: int = 0
    polish: str = "not run"
    started: float = field(default_factory=time.monotonic)
    behavior_started: float = 0.0


class State(TypedDict, total=False):
    plan: dict[str, Any]
    idx: int
    attempts: int  # impl attempts since the last hint
    red_tries: int
    hint: str
    last_run: str
    passed: bool
    red_ok: bool
    satisfied: bool
    phase: Literal["test", "impl", "skipped"]
    waived: Annotated[list[str], operator.add]
    failed: list[str]
    snapshot: dict[str, str]  # Workspace files as they were when this Behavior started: what a skip reverts to
    backup: dict[str, str]  # Workspace files before Polish
    edited: dict[str, str]  # path -> contents, every file the model has written
    test_paths: list[str]
    status: str


SYSTEM = (
    "You are a careful Python programmer working test-first in a small project. "
    "Reply with JSON only: {\"edits\": [{\"path\": ..., \"content\": ...}], \"rationale\": ...}. "
    "Each edit is the FULL new contents of one file, path relative to the project root. No explanations outside the JSON."
)


def build(model: LiveModel, box: Seatbelt, work: Path, rec: Recorder, args: argparse.Namespace) -> StateGraph[State]:
    test_cmd = ".venv/bin/python -m pytest -q --no-header -p no:cacheprovider"

    def note(name: str) -> None:
        rec.trace.append(name)
        print(f"  {name}", flush=True)

    def read(rel: str) -> str:
        path = work / rel
        return path.read_text() if path.is_file() else ""

    def context(state: State) -> str:
        plan = state["plan"]
        shown = dict.fromkeys([*plan.get("files", []), *state.get("edited", {})])
        files = "\n\n".join(
            f"--- {rel} ---\n{trimmed(read(rel), MAX_FILE_LINES)}" for rel in shown if (work / rel).is_file()
        )
        notes = "\n".join(f"- {n}" for n in plan.get("notes", []))
        return (
            f"Goal: {plan['goal']}\nOut of scope: {'; '.join(plan.get('out_of_scope', [])) or 'nothing listed'}\n"
            f"{('Notes:' + chr(10) + notes + chr(10)) if notes else ''}"
            # Later behaviors stay hidden: a model that sees them builds them all now, and their tests then pass at Red.
            f"Behaviors so far, in order:\n" + "\n".join(f"{i + 1}. {b}" for i, b in enumerate(plan["behaviors"][: state.get("idx", 0) + 1]))
            + f"\n\nProject files:\n{file_tree(work)}\n\nCurrent file contents:\n{files or '(none yet)'}"
        )

    def capture() -> dict[str, str]:
        return {rel: read(rel) for rel in file_tree(work).splitlines()}

    def restore(files: dict[str, str]) -> None:
        for rel in file_tree(work).splitlines():
            if rel not in files:
                (work / rel).unlink()
        for rel, text in files.items():
            (work / rel).write_text(text)

    def current(state: State) -> BehaviorStats:
        return rec.behaviors[state["idx"]]

    def apply(state: State, edits: Edits, allow: Literal["tests", "impl"]) -> tuple[dict[str, str], list[str]]:
        """Write the edits that pass the path rules; return (written, refusals)."""
        written: dict[str, str] = {}
        refused: list[str] = []
        if not edits.edits:
            refused.append(f"no edits ({edits.rationale})")
        for edit in edits.edits[:3]:
            try:
                path = resolve_inside(work, edit.path)
                rel = str(path.relative_to(work.resolve()))
                in_tests = rel.startswith("tests/")
                if (allow == "tests") != in_tests:
                    raise ValueError(f"{'only files under tests/' if allow == 'tests' else 'not files under tests/'} may be edited now: {edit.path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(edit.content)
                written[rel] = edit.content
            except ValueError as exc:
                refused.append(str(exc))
        return written, refused

    async def pytest(target: str = "") -> RunResult:
        return await box.run(f"{test_cmd} {target}".strip())

    # --- Stage 1: assumption check
    async def check_assumptions(state: State) -> dict[str, Any]:
        note("check_assumptions")
        failed = []
        for a in state["plan"]["assumptions"]:
            if a["text"] in state.get("waived", []):
                print(f"    ~ {a['text']} (waived)")
            elif not a.get("probe"):
                print(f"    ✓ {a['text']} (no probe)")
            else:
                r = await box.run(a["probe"], 30)
                print(f"    {'✓' if r.code == 0 else '✗'} {a['text']}  [{a['probe']}]")
                if r.code != 0:
                    failed.append(a["text"])
        return {"failed": failed}

    def after_check(state: State) -> Literal["clarify", "write_test"]:
        return "clarify" if state["failed"] or state["plan"].get("open_questions") else "write_test"

    def clarify(state: State) -> dict[str, Any]:
        note("clarify")
        if args.no_hints:  # baseline mode: waive and move on
            plan = {**state["plan"], "open_questions": [], "notes": [*state["plan"].get("notes", []), "waived (no-hints run)"]}
            return {"waived": state["failed"], "plan": plan}
        answer = str(interrupt({"kind": "clarify", "failed": state["failed"], "open_questions": state["plan"].get("open_questions", [])}))
        if answer.lower().startswith("abort"):
            return {"status": "aborted"}
        plan = {**state["plan"], "open_questions": [], "notes": [*state["plan"].get("notes", []), answer]}
        return {"waived": state["failed"], "plan": plan}

    def after_clarify(state: State) -> Literal["check_assumptions", "__end__"]:
        return END if state.get("status") == "aborted" else "check_assumptions"

    # --- Stage 2: make it work
    async def write_test(state: State) -> dict[str, Any]:
        note("write_test")
        idx = state.get("idx", 0)
        if idx >= len(rec.behaviors):
            rec.behaviors.append(BehaviorStats(state["plan"]["behaviors"][idx]))
        fresh = state.get("phase") != "test"  # not a retry of a rejected test
        if fresh:
            rec.behavior_started = time.monotonic()
        behavior = state["plan"]["behaviors"][idx]
        prompt = (
            f"{context(state)}\n\nTask: write ONE new pytest test file, tests/test_behavior_{idx + 1}.py, for this behavior only:\n"
            f"{behavior}\n\nImport from the module as in the existing files; do not change the implementation. "
            "The test must fail until the behavior is implemented."
        )
        if state.get("hint"):
            prompt += f"\n\nA human hint: {state['hint']}"
        if state.get("phase") == "test" and state.get("last_run"):
            prompt += f"\n\nYour previous test was rejected. Output of running it:\n{state['last_run']}"
        edits = await model.edits(SYSTEM, prompt)
        written, refused = apply(state, edits, "tests")
        out: dict[str, Any] = {"idx": idx, "phase": "test", "test_paths": [*written]}
        if fresh:
            out.update(snapshot=capture(), red_tries=0, satisfied=False)
        if refused:
            print(f"    refused: {refused}")
            out["last_run"] = "Edit refused: " + "; ".join(refused)
        return {**out, "edited": {**state.get("edited", {}), **written}}

    async def run_red(state: State) -> dict[str, Any]:
        note("run_red")
        tries = state.get("red_tries", 0)
        if not state.get("test_paths"):
            current(state).red_rejections += 1
            return {"red_ok": False, "red_tries": tries + 1}
        r = await pytest(" ".join(state["test_paths"]))
        import_error = r.code == 2 and ("ModuleNotFoundError" in r.output or "ImportError" in r.output) and "SyntaxError" not in r.output
        red_ok = r.code == 1 or import_error
        print(f"    red: {'failed as it should' if red_ok else f'REJECTED (pytest exit {r.code})'}")
        if red_ok:
            return {"red_ok": True, "last_run": tail(r.output)}
        if r.code == 0:  # a passing test is not a bad test here: keep it and move on rather than burn retries
            current(state).satisfied = True
            return {"red_ok": False, "satisfied": True}
        current(state).red_rejections += 1
        return {"red_ok": False, "red_tries": tries + 1, "last_run": tail(r.output)}

    def after_red(state: State) -> Literal["write_test", "write_impl", "next_behavior", "stuck"]:
        if state["red_ok"]:
            return "write_impl"
        if state.get("satisfied"):
            return "next_behavior"
        return "stuck" if state["red_tries"] >= MAX_ATTEMPTS else "write_test"

    async def write_impl(state: State) -> dict[str, Any]:
        note("write_impl")
        stats = current(state)
        stats.attempts += 1
        attempts = state.get("attempts", 0) + 1
        behavior = state["plan"]["behaviors"][state["idx"]]
        prompt = (
            f"{context(state)}\n\nTask: make the tests pass by editing the implementation (not the tests). "
            f"Current behavior: {behavior}\nWrite the SMALLEST change that makes the tests pass. Do not handle cases no test covers yet; "
            "later behaviors will get their own tests. Keep all earlier behaviors working.\n\n"
            f"Last pytest output (tail):\n{state.get('last_run', '')}"
        )
        if state.get("hint"):
            prompt += f"\n\nA human hint: {state['hint']}"
        edits = await model.edits(SYSTEM, prompt)
        written, refused = apply(state, edits, "impl")
        update: dict[str, Any] = {"attempts": attempts, "phase": "impl", "edited": {**state.get("edited", {}), **written}}
        if refused:
            print(f"    refused: {refused}")
            update["last_run"] = "Edit refused: " + "; ".join(refused)
        return update

    async def run_green(state: State) -> dict[str, Any]:
        note("run_green")
        if state.get("last_run", "").startswith("Edit refused"):
            return {"passed": False}
        r = await pytest()
        print(f"    green: {'all tests pass' if r.code == 0 else f'FAILED (exit {r.code})'}")
        return {"passed": r.code == 0, "last_run": tail(r.output)}

    def after_green(state: State) -> Literal["next_behavior", "write_impl", "stuck"]:
        if state["passed"]:
            return "next_behavior"
        return "write_impl" if state["attempts"] < MAX_ATTEMPTS else "stuck"

    def stuck(state: State) -> dict[str, Any]:
        note("stuck")
        rec.stuck += 1
        stats = current(state)
        answer = "skip" if args.no_hints else str(interrupt({"kind": "stuck", "behavior": stats.behavior, "last_run": state.get("last_run", "")}))
        if answer.lower().startswith("abort"):
            return {"status": "aborted"}
        if answer.lower().startswith("skip"):  # mark failed and put the Workspace back as this Behavior found it
            stats.green, stats.note = False, "stuck"
            stats.seconds = time.monotonic() - rec.behavior_started
            restore(state["snapshot"])
            return {"phase": "skipped", "idx": state["idx"] + 1, "attempts": 0, "hint": "", "last_run": "", "edited": {}}
        stats.hints += 1
        return {"attempts": 0, "red_tries": 0, "hint": answer}

    def after_stuck(state: State) -> Literal["write_test", "write_impl", "sign_off", "__end__"]:
        if state.get("status") == "aborted":
            return END
        if state.get("phase") == "skipped":
            return "write_test" if state["idx"] < len(state["plan"]["behaviors"]) else "sign_off"
        return "write_test" if state.get("phase") == "test" else "write_impl"

    def next_behavior(state: State) -> dict[str, Any]:
        note("next_behavior")
        stats = current(state)
        stats.green = True
        stats.seconds = time.monotonic() - rec.behavior_started
        return {"idx": state["idx"] + 1, "attempts": 0, "phase": "impl", "hint": "", "last_run": ""}

    def after_next(state: State) -> Literal["write_test", "sign_off"]:
        return "write_test" if state["idx"] < len(state["plan"]["behaviors"]) else "sign_off"

    # --- Stage 3: make it right
    async def sign_off(state: State) -> dict[str, Any]:
        note("sign_off")
        demo = ""
        if state["plan"].get("demo"):
            demo = (await box.run(state["plan"]["demo"], 30)).output
        if args.no_hints:
            print(f"    demo output:\n{demo}")
            return {}
        answer = str(interrupt({"kind": "sign_off", "demo": demo}))
        if answer.lower().startswith("abort"):
            return {"status": "aborted"}
        if answer.lower().startswith(("y", "ok", "looks good")):
            return {}
        return {"plan": {**state["plan"], "behaviors": [*state["plan"]["behaviors"], answer]}, "phase": "impl"}

    def after_sign_off(state: State) -> Literal["polish", "write_test", "__end__"]:
        if state.get("status") == "aborted":
            return END
        return "write_test" if state["idx"] < len(state["plan"]["behaviors"]) else "polish"

    async def polish(state: State) -> dict[str, Any]:
        note("polish")
        backup = capture()
        prompt = (
            f"{context(state)}\n\nTask: polish the implementation. Change structure and names (clearer names, type hints, "
            "a short docstring, small helper functions), NEVER behavior. Do not touch tests."
        )
        edits = await model.edits(SYSTEM, prompt)
        _, refused = apply(state, edits, "impl")
        if refused:
            print(f"    refused: {refused}")
        return {"backup": backup}

    async def verify_polish(state: State) -> dict[str, Any]:
        note("verify_polish")
        r = await pytest()
        if r.code == 0:
            rec.polish = "kept"
            print("    polish kept: tests still pass")
        else:
            rec.polish = "reverted"
            restore(state["backup"])
            print("    polish reverted: it broke the suite")
        return {"status": "done"}

    g = StateGraph(State)
    for name, fn in [
        ("check_assumptions", check_assumptions), ("clarify", clarify), ("write_test", write_test), ("run_red", run_red),
        ("write_impl", write_impl), ("run_green", run_green), ("stuck", stuck), ("next_behavior", next_behavior),
        ("sign_off", sign_off), ("polish", polish), ("verify_polish", verify_polish),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "check_assumptions")
    g.add_conditional_edges("check_assumptions", after_check, ["clarify", "write_test"])
    g.add_conditional_edges("clarify", after_clarify, ["check_assumptions", END])
    g.add_edge("write_test", "run_red")
    g.add_conditional_edges("run_red", after_red, ["write_test", "write_impl", "next_behavior", "stuck"])
    g.add_edge("write_impl", "run_green")
    g.add_conditional_edges("run_green", after_green, ["next_behavior", "write_impl", "stuck"])
    g.add_conditional_edges("stuck", after_stuck, ["write_test", "write_impl", "sign_off", END])
    g.add_conditional_edges("next_behavior", after_next, ["write_test", "sign_off"])
    g.add_conditional_edges("sign_off", after_sign_off, ["polish", "write_test", END])
    g.add_edge("polish", "verify_polish")
    g.add_edge("verify_polish", END)
    return g


# --- Driver --------------------------------------------------------------------------------------------------

def show(payload: dict[str, Any]) -> None:
    print(f"\n  ┌─ HUMAN: {payload['kind']}")
    if payload["kind"] == "clarify":
        for f in payload["failed"]:
            print(f"  │ assumption failed: {f}")
        for q in payload["open_questions"]:
            print(f"  │ open question:     {q}")
        print("  └─ <how to proceed> | abort")
    elif payload["kind"] == "stuck":
        print(f"  │ stuck on: {payload['behavior']}")
        print("\n".join(f"  │   {line}" for line in payload["last_run"].splitlines()[-8:]))
        print("  └─ <a hint for the model> | skip | abort")
    else:
        print("  │ demo output:")
        print("\n".join(f"  │   {line}" for line in payload["demo"].splitlines()))
        print("  └─ yes | <what's wrong or missing; becomes a new behavior> | abort")


async def amain(args: argparse.Namespace) -> None:
    kata = Path(args.kata).expanduser().resolve()
    plan = json.loads(Path(args.plan or kata / "plan.json").expanduser().read_text())
    client = OllamaClient(args.host, keep_alive="10m")
    try:
        pre = await run_preflight(client, [Candidate(args.model, args.num_ctx)], env=os.environ, gpu_limit=system_gpu_limit())
        print(f"model {args.model} pinned at num_ctx {pre.num_ctx}")
        work = make_workspace(kata)
        print(f"kata: {kata.name}  workspace: {work}")
        box = Seatbelt(work)
        rec = Recorder()
        model = LiveModel(client, args.model)
        graph = build(model, box, work, rec, args).compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "exec-live"}, "recursion_limit": 200}
        run_input: Any = {"plan": plan, "idx": 0, "attempts": 0, "edited": {}, "snapshot": {}}
        while True:
            pending = None
            async for update in graph.astream(run_input, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    pending = update["__interrupt__"][0].value
            if pending is None:
                break
            show(pending)
            answer = await asyncio.to_thread(input, "  > ")
            run_input = Command(resume=answer)
        final = graph.get_state(config).values
        summary = write_summary(kata, args, work, rec, model, final)
        green = sum(b.green for b in rec.behaviors)
        print(f"\n{green}/{len(plan['behaviors'])} behaviors green; polish {rec.polish}; summary: {summary}")
    finally:
        if args.unload:
            await client.unload(args.model)
        await client.aclose()


def write_summary(kata: Path, args: argparse.Namespace, work: Path, rec: Recorder, model: LiveModel, final: dict[str, Any]) -> Path:
    runs = kata / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs / f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', args.model)}.json"
    path.write_text(json.dumps({
        "model": args.model, "kata": kata.name, "no_hints": args.no_hints, "status": final.get("status"),
        "behaviors": [asdict(b) for b in rec.behaviors],
        "planned": len(final["plan"]["behaviors"]),
        "totals": {
            "green": sum(b.green for b in rec.behaviors), "attempts": sum(b.attempts for b in rec.behaviors),
            "red_rejections": sum(b.red_rejections for b in rec.behaviors), "hints": sum(b.hints for b in rec.behaviors),
            "satisfied": sum(b.satisfied for b in rec.behaviors),
            "seconds": round(time.monotonic() - rec.started, 1), "model_seconds": round(model.model_seconds, 1),
            "model_calls": model.calls,
        },
        "stuck": rec.stuck, "polish": rec.polish, "trace": rec.trace, "workspace": str(work),
    }, indent=2) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kata", help="a kata repo with a plan.json")
    p.add_argument("--plan", help="Plan JSON (default <kata>/plan.json)")
    p.add_argument("--model", default="gemma4:e4b")
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--no-hints", action="store_true", help="baseline: Stuck marks the Behavior failed, clarify waives, Sign-off says yes")
    p.add_argument("--unload", action="store_true", help="unload the model at the end (use between models)")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
