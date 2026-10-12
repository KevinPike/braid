"""What the four F13 runners share, so that only the framework differs: the kata copy, the two Roles' instructions, the
file and test tools each framework wraps in its own tool type, and the one JSON line a runner prints.

Isolation (README): runners 2-4 run model-written code (pytest on the model's module and tests) here, outside braid's
Sandbox and outside ADR 0005. What holds instead:

- every run works in a fresh temporary copy of the kata repo (``GOAL.md``, a stub module, ``pyproject.toml``,
  ``tests/``), never in the project or in ``bench/exotic``, so no Hidden test, reference or mutant is reachable;
- the file tools refuse any path outside that copy, and each Role may write only its own files (the tester test
  files, the implementer everything else);
- pytest runs with ``env_without_keys()``: no ``ANTHROPIC_*`` variable, no ``*_API_KEY``, ``*_TOKEN`` or cloud
  credential reaches the model's code.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
EXOTIC = ROOT / "bench" / "exotic"
DEFAULT_MODEL = "claude-haiku-5-5"
TEST_TIMEOUT_S = 120.0
MAX_TOOL_OUTPUT = 6000  # characters of pytest output handed back to a model

Role = Literal["tester", "implementer"]

# A key-like variable never reaches the model's code (pytest, and anything a test imports).
SECRET = re.compile(r"(^ANTHROPIC_|^CLAUDE_CODE_OAUTH|^AWS_|^GOOGLE_|^AZURE_|API_KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)


def module_name(kata: str) -> str:
    return kata.replace("-", "_")


def make_katas() -> Any:
    """``examples/graphs/make_katas.py``, which writes a kata repo from ``GOAL.md`` and ``summary.txt`` only."""
    path = ROOT / "examples" / "graphs" / "make_katas.py"
    spec = importlib.util.spec_from_file_location("make_katas", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kata_repo(root: Path, kata: str, *, sync: bool = False) -> Path:
    """The kata repo ``make_katas.py`` creates, under ``root``. Its chatter goes to stderr: stdout is the JSON line."""
    mk = make_katas()
    katas = mk.exotic_katas()
    if kata not in katas:
        raise SystemExit(f"{kata!r} is not an exotic kata (known: {', '.join(sorted(katas))})")
    summary, goal_md = katas[kata]
    with contextlib.redirect_stdout(sys.stderr):
        mk.make(root, kata, summary, goal_md, sync=sync)
    return root / kata


@contextlib.contextmanager
def kata_copy(kata: str) -> Iterator[Path]:
    """A fresh temporary copy of the kata repo, removed afterwards."""
    with tempfile.TemporaryDirectory(prefix=f"braid-f13-{kata}-") as tmp:
        yield kata_repo(Path(tmp).resolve(), kata)


def export(work: Path, out: Path) -> list[str]:
    """Copy the run's files (no ``.git``, ``.venv``, caches) into ``out``; returns them, relative."""
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(work.rglob("*")):
        rel = path.relative_to(work)
        if any(p.startswith(".") or p in ("__pycache__", "runs") for p in rel.parts[:-1]) or not path.is_file() or path.is_symlink():
            continue
        if path.suffix == ".pyc":
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out / rel)
        copied.append(rel.as_posix())
    return copied


def write_trace(out: Path, name: str, data: Any) -> Path:
    """A runner's own record of the run, under ``<out>/.trace/`` (the grader skips dot-directories)."""
    path = out / ".trace" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2, default=str) + "\n")
    return path


# --- the two Roles ---------------------------------------------------------------------------------------------------

def tester_prompt(kata: str) -> str:
    module = module_name(kata)
    return f"""You are the tester in a two-role test-driven team working on the kata in the current directory.
Read GOAL.md: it is the whole specification. Write pytest tests in tests/test_{module}.py that pin down every rule
GOAL.md states: each numbered rule, the boundaries between bands, ties, empty input, and every error GOAL.md names.
Derive each expected value from GOAL.md by hand, never from the implementation. Import the code under test with
`from {module} import ...`. You may write only files under tests/. Do not write {module}.py.
Run the tests once: they are expected to fail, since the implementation is a stub. Then hand the work to the
implementer, naming the test file. If the implementer hands back a test it believes contradicts GOAL.md, re-read
GOAL.md, fix the test if it is wrong, and hand back again."""


def implementer_prompt(kata: str) -> str:
    module = module_name(kata)
    return f"""You are the implementer in a two-role test-driven team working on the kata in the current directory.
Read GOAL.md and the tester's tests under tests/. Write {module}.py so that it meets GOAL.md and every test passes.
You may not edit any test file. Run the tests after each change. If a test contradicts GOAL.md, say exactly which
rule it contradicts and hand it back to the tester (at most twice in the whole run). Finish when every test passes,
with a one-line summary."""


TASK = ("Build the kata in the current directory test-first: the tester writes tests from GOAL.md, then the "
        "implementer makes them pass. GOAL.md is the specification.")


# --- tools, shared by every framework's own tool type ----------------------------------------------------------------

class ToolRefused(ValueError):
    """A tool call the run does not allow: a path outside the kata copy, or a file the Role may not write."""


def resolve(work: Path, rel: str) -> Path:
    path = (work / rel).resolve()
    if path != work and work not in path.parents:
        raise ToolRefused(f"{rel!r} is outside the kata directory")
    if any(p.startswith(".") for p in path.relative_to(work).parts):
        raise ToolRefused(f"{rel!r} is in a hidden directory")
    return path


def is_test_path(rel: str) -> bool:
    parts = Path(rel).parts
    name = parts[-1] if parts else ""
    return (bool(parts) and parts[0] in ("tests", "test")) or name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


def read_file(work: Path, rel: str) -> str:
    path = resolve(work, rel)
    if not path.is_file():
        raise ToolRefused(f"{rel!r} does not exist")
    return path.read_text()


def write_file(work: Path, role: Role, rel: str, content: str) -> str:
    path = resolve(work, rel)
    rel = path.relative_to(work).as_posix()
    if role == "tester" and not is_test_path(rel):
        raise ToolRefused(f"the tester may write only test files, not {rel!r}")
    if role == "implementer" and is_test_path(rel):
        raise ToolRefused(f"the implementer may not write test files ({rel!r})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"wrote {rel} ({len(content)} characters)"


def list_files(work: Path) -> str:
    return "\n".join(
        p.relative_to(work).as_posix() for p in sorted(work.rglob("*"))
        if p.is_file() and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(work).parts)
    )


def env_without_keys() -> dict[str, str]:
    """The environment the model's code runs with: this one, minus every credential-like variable."""
    env = {k: v for k, v in os.environ.items() if not SECRET.search(k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_tests(work: Path, target: str = "") -> str:
    """pytest in the kata copy, with no credentials in its environment; the exit code and the tail of the output."""
    command = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if target:
        command.append(str(resolve(work, target).relative_to(work)))
    try:
        done = subprocess.run(command, cwd=work, env=env_without_keys(), capture_output=True, text=True, timeout=TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"pytest timed out after {TEST_TIMEOUT_S:.0f}s"
    output = (done.stdout + done.stderr)[-MAX_TOOL_OUTPUT:]
    return f"exit code {done.returncode}\n{output}"


# --- the JSON line ---------------------------------------------------------------------------------------------------

@dataclass
class Usage:
    """What a runner reports: tokens, calls and cost, in total and per Role where the framework tells them apart."""

    framework: str
    kata: str
    model: str
    out: str
    calls: int = 0
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float | None = None  # None when the framework gives no way to know it
    cost_source: str = ""  # how cost_usd was found
    roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    seconds: float = 0.0
    status: str = "ok"
    trace: str = ""  # where the run's trace lives
    notes: list[str] = field(default_factory=list)


def price(model: str, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int) -> float | None:
    """braid's own prices (``harness.usage.PRICING``), or None for a model it has no price for."""
    try:
        from harness.usage import PRICING, api_call
    except ImportError:
        return None
    if model not in PRICING:
        return None
    return api_call(model, input_tokens, output_tokens, cache_read, cache_write, None).cost_usd


def report(usage: Usage) -> None:
    """The one line a runner prints on stdout."""
    data = asdict(usage)
    if data["cost_usd"] is not None:
        data["cost_usd"] = round(data["cost_usd"], 6)
    print(json.dumps(data), flush=True)


def no_sampling(body: dict[str, Any]) -> None:
    """Claude 5.5 models answer 400 to ``temperature``, ``top_p`` and ``top_k``: fail before sending one."""
    sent = sorted(k for k in ("temperature", "top_p", "top_k") if body.get(k) is not None)
    if sent:
        raise RuntimeError(f"refusing to send {', '.join(sent)} to a Claude 5.5 model")
