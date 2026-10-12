"""F13 grading adapter: a runner's output directory -> F7's hidden grade.

    uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/strands [--sandbox local]

Prints one JSON line: ``hidden_passed``, ``hidden_total``, ``mutants_killed``, ``mutants_total`` (F7's ``Grade``) and
``own_tests``, the number of test files the runner wrote. Mutants are killed only by the runner's own tests, so with no
test files ``mutants_killed`` is 0 and says nothing.

This reuses ``harness.workflow.bench.grade`` unchanged. That function grades a Workspace, so a plain directory is
wrapped as one (``DirWorkspace``): a staging copy in the system temp dir, with the output's files (no dot-directories,
no caches) and a ``.venv`` link to a venv that has pytest (default: the one running this script). Grading then runs in
scratch copies of that, in the Sandbox (Seatbelt on macOS), exactly as ``harness-workflow bench`` grades a run. The
output directory itself is never written.

When the output has no ``pyproject.toml``, the staging copy gets the kata repo's, whose ``pythonpath = ["."]`` lets a
test under ``tests/`` import the module, as it does in a kata repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.sandbox import LocalSandbox, Sandbox, SeatbeltSandbox, Workspace
import harness.workflow.bench as bench

EXOTIC = bench.HARNESS_ROOT / "bench" / "exotic"
PYPROJECT = '[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n\n' \
            '[tool.pytest.ini_options]\npythonpath = ["."]\n'
TEST_COMMAND = ".venv/bin/python -m pytest -q --no-header -p no:cacheprovider"
SKIPPED_DIRS = {"__pycache__", "node_modules"}  # and every directory whose name starts with "."


def output_files(out: Path) -> list[str]:
    """What a runner left in ``out``, relative and sorted: no dot-directories (``.git``, ``.venv``, ``.trace``), no
    caches, no symlinks."""
    found = []
    for path in out.rglob("*"):
        rel = path.relative_to(out)
        if any(p.startswith(".") or p in SKIPPED_DIRS for p in rel.parts[:-1]):
            continue
        if path.is_file() and not path.is_symlink() and path.suffix != ".pyc":
            found.append(rel.as_posix())
    return sorted(found)


class DirWorkspace(Workspace):
    """A plain directory standing in for a Workspace, for ``bench.grade``: only ``path`` and ``list_files`` are used.
    It has no git, so ``changes``, ``apply`` and ``remove`` must not be called."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, path, path / ".git")

    async def list_files(self) -> list[str]:
        return output_files(self.path)


@dataclass(frozen=True)
class FrameworkGrade:
    grade: bench.Grade
    own_tests: int  # test files the runner wrote (F7's rule): without any, mutants_killed means nothing

    def fields(self) -> dict[str, Any]:
        return {**self.grade.fields(), "own_tests": self.own_tests}


def kata_dir(kata: str) -> Path:
    path = EXOTIC / kata
    if not (path / "hidden" / bench.HIDDEN_FILE).is_file():
        known = ", ".join(sorted(d.name for d in EXOTIC.iterdir() if (d / "hidden").is_dir()))
        raise ValueError(f"{kata!r} is not an exotic kata (known: {known})")
    return path


async def grade_dir(
    out: Path, kata: str, *, sandbox: Sandbox, venv: Path | None = None, test_command: str = TEST_COMMAND,
    timeout_s: float = 120.0,
) -> FrameworkGrade:
    """Grade the kata module and tests in ``out`` by the kata's Hidden tests and mutants. A missing or broken module
    scores 0 of the hidden suite's size (collected against the reference), not 0/0."""
    hidden = kata_dir(kata) / "hidden"
    module = kata.replace("-", "_")
    venv = (venv or Path(sys.prefix)).resolve()
    files = output_files(out) if out.is_dir() else []
    with tempfile.TemporaryDirectory(prefix="braid-f13-stage-") as tmp:
        stage = Path(tmp).resolve()
        for rel in files:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / rel, stage / rel)
        if not (stage / "pyproject.toml").exists():
            (stage / "pyproject.toml").write_text(PYPROJECT.format(name=kata))
        (stage / ".venv").symlink_to(venv)
        graded = await bench.grade(
            DirWorkspace(stage), hidden, module, sandbox=sandbox, test_command=test_command, timeout_s=timeout_s,
        )
    return FrameworkGrade(graded, sum(bench.is_test_file(f) and f.endswith(".py") for f in files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--kata", required=True, help="an exotic kata, e.g. ferry-fare")
    parser.add_argument("--out", required=True, type=Path, help="the runner's output directory")
    parser.add_argument("--sandbox", choices=("seatbelt", "local"), default="seatbelt" if sys.platform == "darwin" else "local")
    parser.add_argument("--venv", type=Path, default=None, help="a venv with pytest (default: this interpreter's)")
    args = parser.parse_args()
    sandbox: Sandbox = SeatbeltSandbox() if args.sandbox == "seatbelt" else LocalSandbox()
    result = asyncio.run(grade_dir(args.out, args.kata, sandbox=sandbox, venv=args.venv))
    print(json.dumps({"kata": args.kata, "out": str(args.out), **result.fields()}))


if __name__ == "__main__":
    main()
