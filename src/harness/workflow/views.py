"""F11 Views: what a Role is shown of the Workspace. ``full`` is the files as they are; ``interface`` (the black-box
tester) shows an implementation file as its public interface only: signatures, docstrings and type hints, no bodies.

Built with ``ast``, never by running the file. A file that does not parse shows nothing of its contents (a note says
so), so a broken file cannot leak its bodies. pytest's tracebacks quote the implementation's lines, so output the tester
reads is passed through ``redact`` too (``examples/graphs/roles_proto.py`` found the leak).
"""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

HIDDEN = "[implementation line hidden]"


def _dots() -> ast.stmt:
    return ast.Expr(ast.Constant(...))


def is_test(rel: str) -> bool:
    """A test file, as F7's grading names one: under ``tests/`` or ``test/``, ``test_*.py``, ``*_test.py``, ``conftest.py``."""
    p = PurePosixPath(rel)
    return bool({"tests", "test"} & set(p.parts[:-1])) or p.name.startswith("test_") or p.name.endswith("_test.py") or (
        p.name == "conftest.py"
    )


def is_implementation(rel: str) -> bool:
    return rel.endswith(".py") and not is_test(rel)


def _public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _doc(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    doc = ast.get_docstring(node, clean=False)
    return [ast.Expr(ast.Constant(doc))] if doc is not None else []


def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
    """The public names of a module or class body, with every function body and assigned value replaced by ``...``."""
    kept: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _public(node.name):
            node.body = [*_doc(node), _dots()]
            kept.append(node)
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            node.body = [*_doc(node), *_strip(node.body)] or [_dots()]
            kept.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and _public(node.target.id):
            kept.append(ast.AnnAssign(node.target, node.annotation, ast.Constant(...) if node.value else None, node.simple))
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and _public(t.id) for t in node.targets):
            kept.append(ast.Assign(node.targets, ast.Constant(...)))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            kept.append(node)
    return kept


def interface(source: str) -> str:
    """``source``'s public interface: imports, signatures with type hints and decorators, docstrings, and public names
    assigned at module or class level (their values as ``...``). Private names (``_x``, not dunders) are left out.
    Raises ``SyntaxError`` for a file that does not parse."""
    tree = ast.parse(source)
    tree.body = [*_doc(tree), *_strip(tree.body)]
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def shown(rel: str, text: str) -> str:
    """A file as the ``interface`` View shows it: an implementation file as its interface, any other file as it is."""
    if not is_implementation(rel):
        return text
    try:
        return interface(text)
    except (SyntaxError, ValueError) as exc:  # ValueError: a null byte
        line = f" (line {exc.lineno})" if isinstance(exc, SyntaxError) and exc.lineno else ""
        return f"# This file does not parse{line}, so its interface cannot be shown.\n"


def body_lines(files: dict[str, str]) -> set[str]:
    """Every stripped line inside a function body of an implementation file (docstrings aside): what ``redact`` hides.
    A file that does not parse gives all its lines, as it shows none."""
    found: set[str] = set()
    for rel, text in files.items():
        if not is_implementation(rel):
            continue
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            found |= {ln.strip() for ln in lines}
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = node.body[1:] if _doc(node) else node.body
                if body:
                    found |= {ln.strip() for ln in lines[body[0].lineno - 1: node.end_lineno]}
    return {ln for ln in found if len(ln) > 3}  # not ``)``, ``]`` or ``pass``


def redact(output: str, hidden: set[str]) -> str:
    """``output`` with every line that quotes one of ``hidden`` (pytest marks them with ``>`` or ``E``) replaced."""
    if not hidden:
        return output
    return "\n".join(
        f"    {HIDDEN}" if re.sub(r"^\s*(>|E\s)?\s*", "", line).strip() in hidden else line for line in output.split("\n")
    )
