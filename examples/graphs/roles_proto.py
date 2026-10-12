"""F11 prototype: what does a black-box tester see, and does anything leak a function body into its prompt?

    uv run python examples/graphs/roles_proto.py

1. The ``interface`` view of every exotic kata's reference implementation, built with ``ast``: signatures, docstrings
   and type hints, no bodies. Checks that no body line survives, and how much smaller the view is.
2. A test that fails inside the implementation, run with pytest: its traceback quotes the implementation's lines, and
   the tester sees pytest's output when its test is rejected. Shows that output before and after redaction.
3. A scripted two-Role exchange with a model per Role, counting requests per model.

No model runs: the "models" are scripted.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

KATAS = Path(__file__).resolve().parents[2] / "bench" / "exotic"


def interface(source: str) -> str:
    """The public interface of a module: bodies become ``...``, private names go, docstrings stay."""
    tree = ast.parse(source)

    def strip(nodes: list[ast.stmt]) -> list[ast.stmt]:
        kept: list[ast.stmt] = []
        for node in nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_") or (
                isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ):
                doc = ast.get_docstring(node)
                node.body = [*([ast.Expr(ast.Constant(doc))] if doc else []), ast.Expr(ast.Constant(...))]
                kept.append(node)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node)
                node.body = [*([ast.Expr(ast.Constant(doc))] if doc else []), *strip(node.body)] or [ast.Expr(ast.Constant(...))]
                kept.append(node)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                kept.append(node)
        return kept

    doc = ast.get_docstring(tree)
    tree.body = [*([ast.Expr(ast.Constant(doc))] if doc else []), *strip(tree.body)]
    return ast.unparse(tree)


def body_lines(source: str) -> set[str]:
    """Every stripped source line inside a function body, docstrings aside: what must never reach the tester."""
    lines = source.splitlines()
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = node.body[1 if ast.get_docstring(node) else 0].lineno if len(node.body) > bool(ast.get_docstring(node)) else None
            if start is not None:
                found |= {ln.strip() for ln in lines[start - 1: node.end_lineno] if len(ln.strip()) > 3}
    return found


def redact(output: str, hidden: set[str]) -> str:
    return "\n".join(
        "    [implementation line hidden]" if line.lstrip(">E ").strip() in hidden else line for line in output.splitlines()
    )


def part_one() -> None:
    print("== 1. the interface view of each exotic reference ==")
    for ref in sorted(KATAS.glob("*/reference/*.py")):
        source = ref.read_text()
        view = interface(source)
        leaked = [ln for ln in body_lines(source) if ln in {v.strip() for v in view.splitlines()}]
        print(f"{ref.parent.parent.name:15} {len(source):5} -> {len(view):4} chars; body lines leaked: {len(leaked)}")
    print("\nration-queue, as the tester sees it:\n" + interface((KATAS / "ration-queue/reference/ration_queue.py").read_text()))


def part_two() -> None:
    print("\n== 2. a rejected test's pytest output quotes the implementation ==")
    source = (KATAS / "ration-queue/reference/ration_queue.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "ration_queue.py").write_text(source)
        (Path(tmp) / "test_x.py").write_text(
            "from ration_queue import distribute\n\n\ndef test_x() -> None:\n    assert distribute([('a', 1)], -1) == {}\n")
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"], cwd=tmp,
                             capture_output=True, text=True)
    hidden = body_lines(source)
    leaks = [ln for ln in run.stdout.splitlines() if ln.lstrip(">E ").strip() in hidden]
    print(f"{len(leaks)} implementation lines in the raw output, e.g. {leaks[:2]}")
    after = redact(run.stdout, hidden)
    print(f"after redaction: {sum(ln.lstrip('>E ').strip() in hidden for ln in after.splitlines())} left")


class Scripted:
    def __init__(self, name: str, replies: list[str], sent: Counter[str]) -> None:
        self.name, self.replies, self.sent = name, replies, sent

    async def complete(self, system: str, prompt: str) -> str:
        self.sent[self.name] += 1
        return self.replies.pop(0)


async def part_three() -> None:
    print("\n== 3. a model per Role ==")
    sent: Counter[str] = Counter()
    roles = {"tester": Scripted("sonnet", ["test 1", "test 2"], sent), "implementer": Scripted("haiku", ["impl"] * 3, sent)}
    for _behavior in range(2):
        await roles["tester"].complete("TESTER_SYSTEM", "interface view")
        await roles["implementer"].complete("SYSTEM", "full view")
    await roles["implementer"].complete("SYSTEM", "polish")
    print(dict(sent), "-- the Polish call is the implementer's")


if __name__ == "__main__":
    part_one()
    part_two()
    asyncio.run(part_three())
