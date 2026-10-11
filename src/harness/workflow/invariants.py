"""F3: Plan Invariants, and the asserted examples they are checked against.

A Dispute (an implementer saying a test is wrong) never routes by itself (ADR 0004); code decides, by evaluating the
Plan's Invariants on every example a test asserts. The extractor here is deterministic: it reads a test file with
``ast`` and returns ``assert fn(...) == expected`` (either way round) as (test, fn, args, kwargs, expected), including
the rows of a ``pytest.mark.parametrize`` table. Values are evaluated exactly: literals, plus ``+``, ``-`` and ``*``
over values already known, and names bound by the parametrize table or by an earlier ``name = <value>`` in the test.
Anything else (a call, an attribute, a comprehension) makes the assert unextractable: it is skipped and counted, never
guessed. The Invariants themselves are model- or human-written Python, so they are evaluated in the Sandbox
(``CHECKER``), never in the harness.
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

MAX_SEQUENCE = 10_000  # a folded list or string longer than this is treated as unextractable


class Invariant(BaseModel):
    """A rule from the Plan: ``check`` is a Python expression over one call of the function under test, with ``fn`` (its
    name), ``args``, ``kwargs`` and ``result`` bound; it must be true for every expected value a test asserts.
    ``fn``, when set, limits the rule to calls of that function."""

    text: str
    check: str
    fn: str | None = None


@dataclass(frozen=True)
class Example:
    test: str  # "path::test_name" or "path::test_name[row]"
    fn: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expected: Any
    line: int


@dataclass
class Extraction:
    examples: list[Example] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # "path::test line N: why"


class Unknown(Exception):
    """A value the extractor will not evaluate."""


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
}


def evaluate(node: ast.expr, names: Mapping[str, Any]) -> Any:
    """The exact value of ``node``: ``ast.literal_eval`` extended with ``+ - *`` and known names. Raises ``Unknown``."""
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise Unknown(f"name {node.id!r} is not a literal")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = evaluate(node.left, names), evaluate(node.right, names)
        if isinstance(node.op, ast.Mult) and any(isinstance(v, (str, list, tuple)) for v in (left, right)):
            n = right if isinstance(right, int) else left
            if not isinstance(n, int) or n * max(len(v) for v in (left, right) if isinstance(v, (str, list, tuple))) > MAX_SEQUENCE:
                raise Unknown("sequence repetition too large")
        try:
            return _BINOPS[type(node.op)](left, right)
        except TypeError as exc:
            raise Unknown(str(exc)) from exc
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate(node.operand, names)
        if not isinstance(value, (int, float, complex)):
            raise Unknown("unary sign on a non-number")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [evaluate(e, names) for e in node.elts]
        return items if isinstance(node, ast.List) else tuple(items) if isinstance(node, ast.Tuple) else set(items)
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise Unknown("dict unpacking")
        return {evaluate(k, names): evaluate(v, names) for k, v in zip(node.keys, node.values, strict=True) if k is not None}
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise Unknown(f"{type(node).__name__} is not a literal") from exc


def _callee(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):  # module.fn(...)
        return node.attr
    return None


def _call_and_expected(test: ast.expr) -> tuple[ast.Call, ast.expr] | None:
    """``fn(...) == expected`` or ``expected == fn(...)``; None for any other assert."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return None
    left, right = test.left, test.comparators[0]
    if isinstance(left, ast.Call) and _callee(left.func):
        return left, right
    if isinstance(right, ast.Call) and _callee(right.func):
        return right, left
    return None


def _parametrize_rows(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, dict[str, Any]]] | None:
    """The rows of every literal ``pytest.mark.parametrize`` table on ``fn`` (their product, as pytest does), as
    (row id, bindings); ``[("", {})]`` when there is none; None when a table cannot be evaluated."""
    rows: list[tuple[str, dict[str, Any]]] = [("", {})]
    for deco in fn.decorator_list:
        if not (isinstance(deco, ast.Call) and _callee(deco.func) == "parametrize" and len(deco.args) >= 2):
            continue
        try:
            raw_names = evaluate(deco.args[0], {})
            argnames = [n.strip() for n in raw_names.split(",")] if isinstance(raw_names, str) else list(raw_names)
            table: list[tuple[str, dict[str, Any]]] = []
            values_node = deco.args[1]
            if not isinstance(values_node, (ast.List, ast.Tuple)):
                return None
            for i, entry in enumerate(values_node.elts):
                if isinstance(entry, ast.Call) and _callee(entry.func) == "param":  # pytest.param(..., marks=..., id=...)
                    values = [evaluate(a, {}) for a in entry.args]
                else:
                    value = evaluate(entry, {})
                    values = list(value) if len(argnames) > 1 else [value]
                if len(values) != len(argnames):
                    return None
                table.append((str(i), dict(zip(argnames, values, strict=True))))
        except Unknown:
            return None
        rows = [(f"{a}-{b}".strip("-"), {**ra, **rb}) for a, ra in rows for b, rb in table]
    return rows


def _test_functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    yield item


def extract(source: str, path: str = "test.py") -> Extraction:
    """Every literal ``assert fn(...) == expected`` in the test functions of ``source``."""
    out = Extraction()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        out.skipped.append(f"{path}: does not parse ({exc.msg})")
        return out
    for fn in _test_functions(tree):
        rows = _parametrize_rows(fn)
        name = f"{path}::{fn.name}"
        if rows is None:
            out.skipped.append(f"{name}: its parametrize table is not literal")
            continue
        for row_id, bindings in rows:
            test = f"{name}[{row_id}]" if row_id else name
            names = dict(bindings)
            for stmt in fn.body:  # straight-line statements only: a binding inside a branch or loop is not followed
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    try:
                        names[stmt.targets[0].id] = evaluate(stmt.value, names)
                    except Unknown:
                        names.pop(stmt.targets[0].id, None)
                    continue
                for node in ast.walk(stmt):
                    if not isinstance(node, ast.Assert):
                        continue
                    found = _call_and_expected(node.test)
                    if found is None:
                        continue
                    call, expected = found
                    try:
                        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                            raise Unknown("star-arguments")
                        example = Example(
                            test, _callee(call.func) or "?", tuple(evaluate(a, names) for a in call.args),
                            {k.arg: evaluate(k.value, names) for k in call.keywords if k.arg is not None},
                            evaluate(expected, names), node.lineno,
                        )
                    except Unknown as exc:
                        out.skipped.append(f"{test} line {node.lineno}: {exc}")
                        continue
                    out.examples.append(example)
    return out


# --- evaluating Invariants (in the Sandbox) ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    test: str
    example: str  # "fn(args) == expected", as the test asserts it
    invariant: str  # the Invariant's text
    check: str
    value: Any  # what the check returned (False, or a falsy value)


@dataclass(frozen=True)
class Unchecked:
    test: str
    example: str
    invariant: str
    error: str  # the exception the check raised


@dataclass
class CheckResult:
    violations: list[Violation] = field(default_factory=list)
    unchecked: list[Unchecked] = field(default_factory=list)
    checked: int = 0  # (example, invariant) pairs that evaluated true or false
    examples: int = 0
    skipped: list[str] = field(default_factory=list)  # asserts the extractor would not evaluate
    error: str = ""  # the checker itself failed (bad output, timeout)

    def summary(self) -> dict[str, Any]:
        return {
            "examples": self.examples, "checked": self.checked, "skipped": self.skipped, "error": self.error,
            "violations": [v.__dict__ for v in self.violations], "unchecked": [u.__dict__ for u in self.unchecked],
        }


def show_example(e: Example) -> str:
    args = [repr(a) for a in e.args] + [f"{k}={v!r}" for k, v in e.kwargs.items()]
    return f"{e.fn}({', '.join(args)}) == {e.expected!r}"


def checker_input(examples: list[Example], invariants: list[Invariant]) -> str:
    """What the Sandbox checker reads: values as ``repr``, which ``ast.literal_eval`` reads back exactly."""
    return json.dumps({
        "examples": [{"test": e.test, "fn": e.fn, "args": repr(e.args), "kwargs": repr(e.kwargs), "expected": repr(e.expected),
                      "shown": show_example(e)} for e in examples],
        "invariants": [i.model_dump() for i in invariants],
    })


# Runs in the Sandbox with ``python -I``: reads checker_input from argv[1], prints one JSON list of outcomes.
CHECKER = r'''
import ast, json, sys
SAFE = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map", "max",
    "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "TypeError")}
data = json.load(open(sys.argv[1]))
out = []
for ex in data["examples"]:
    args, kwargs, result = (ast.literal_eval(ex[k]) for k in ("args", "kwargs", "expected"))
    for inv in data["invariants"]:
        if inv.get("fn") and inv["fn"] != ex["fn"]:
            continue
        scope = {"__builtins__": SAFE, "fn": ex["fn"], "args": args, "kwargs": kwargs, "result": result}
        row = {"test": ex["test"], "example": ex["shown"], "invariant": inv["text"], "check": inv["check"]}
        try:
            value = eval(inv["check"], scope)
            row["ok"] = bool(value)
            row["value"] = repr(value)[:200]
        except Exception as exc:
            row["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
        out.append(row)
print(json.dumps(out))
'''


def read_checker_output(text: str, examples: int, skipped: list[str]) -> CheckResult:
    result = CheckResult(examples=examples, skipped=skipped)
    try:
        rows = json.loads(text.strip().splitlines()[-1]) if text.strip() else None
    except json.JSONDecodeError:
        rows = None
    if not isinstance(rows, list):
        result.error = "the invariant checker printed no result: " + text[-500:]
        return result
    for row in rows:
        if "error" in row:
            result.unchecked.append(Unchecked(row["test"], row["example"], row["invariant"], row["error"]))
        elif row["ok"]:
            result.checked += 1
        else:
            result.checked += 1
            result.violations.append(Violation(row["test"], row["example"], row["invariant"], row["check"], row["value"]))
    return result


def feedback(violations: list[Violation]) -> str:
    """For ``write_test``: which expected values break which of the Plan's rules."""
    lines = ["The test asserts an expected value that breaks a rule the Plan states, checked by code:"]
    for v in violations[:5]:
        lines.append(f"- {v.test}: `{v.example}` breaks \"{v.invariant}\" (`{v.check}` is {v.value})")
    lines.append("Fix the expected value (or the input) so it follows the Plan's rules; do not drop the case.")
    return "\n".join(lines)


def show_rules(invariants: list[Invariant]) -> str:
    return "\n".join(f"- {i.text}" + (f" (for {i.fn})" if i.fn else "") + f": `{i.check}`" for i in invariants)
