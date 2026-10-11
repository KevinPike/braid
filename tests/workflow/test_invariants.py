"""F3 gates 1 and 6: the asserted-example extractor, and the Sandbox checker's handling of a rule that raises."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from harness.workflow.invariants import CHECKER, CheckResult, Invariant, checker_input, extract, feedback, read_checker_output


def test_plain_asserts_are_found_in_both_orders() -> None:
    found = extract('''
from mod import mul, wrap


def test_one():
    assert mul(2, 3) == 6
    assert "a\\nb" == wrap("a b", width=1)
    assert mul(-2, 3) == -6


class TestGroup:
    def test_in_a_class(self):
        assert mod.mul(4, 5) == 20
''', "tests/t.py")
    got = [(e.test, e.fn, e.args, e.kwargs, e.expected) for e in found.examples]
    assert got == [
        ("tests/t.py::test_one", "mul", (2, 3), {}, 6),
        ("tests/t.py::test_one", "wrap", ("a b",), {"width": 1}, "a\nb"),
        ("tests/t.py::test_one", "mul", (-2, 3), {}, -6),
        ("tests/t.py::test_in_a_class", "mul", (4, 5), {}, 20),
    ]
    assert found.skipped == []


def test_a_single_argname_takes_each_row_whole_as_pytest_does() -> None:
    found = extract('@pytest.mark.parametrize(("pair",), [((1, 2),)])\ndef test_p(pair):\n    assert f(pair) == 3\n')
    assert found.examples[0].args == (((1, 2),),)


def test_parametrized_tables_give_one_example_per_row() -> None:
    found = extract('''
import pytest


@pytest.mark.parametrize("a, b, expected", [(1, 2, 2), (3, 4, 12), pytest.param(0, 9, 0, id="zero")])
def test_mul(a, b, expected):
    assert expected == mul(a, b)


@pytest.mark.parametrize("text", ["x"])
@pytest.mark.parametrize(["n"], [1, 2])
def test_product(text, n):
    assert rep(text, n) == text * n
''')
    assert [(e.test.split("::")[1], e.args, e.expected) for e in found.examples] == [
        ("test_mul[0]", (1, 2), 2), ("test_mul[1]", (3, 4), 12), ("test_mul[2]", (0, 9), 0),
        ("test_product[0-0]", ("x", 1), "x"), ("test_product[0-1]", ("x", 2), "xx"),
    ]


def test_folded_literals_and_earlier_assignments_are_evaluated_exactly() -> None:
    found = extract('''
def test_bowling():
    rolls = [10, 3, 4] + [0] * 16
    assert score(rolls) == 17
    assert score([1, 2] * 10) == 3 + 7 - 1 * 1
''')
    assert [(e.args, e.expected) for e in found.examples] == [(([10, 3, 4] + [0] * 16,), 17), (([1, 2] * 10,), 9)]


def test_non_literal_asserts_are_skipped_and_counted_never_guessed() -> None:
    found = extract('''
def helper():
    return 3


def test_mixed():
    assert mul(helper(), 2) == 6
    assert mul(2, 3) == helper()
    assert mul(*[2, 3]) == 6
    assert mul(x, 3) == 6
    assert [i for i in range(3)] == mul(1, 1)
    assert mul(2, 3) == 6
    assert mul(2, 3) != 7
    assert "x" * 100000 == mul(1, 1)


def test_dynamic_table(a):
    pass


@pytest.mark.parametrize("a", make_rows())
def test_rows(a):
    assert f(a) == 1
''')
    assert [(e.args, e.expected) for e in found.examples] == [((2, 3), 6)]
    assert len(found.skipped) == 7  # six unevaluable asserts in test_mixed, plus the non-literal table
    assert any("helper" in s or "Call" in s for s in found.skipped)
    assert any("test_rows" in s and "parametrize" in s for s in found.skipped)


def test_a_file_that_does_not_parse_is_reported() -> None:
    found = extract("def test_x(:\n", "tests/bad.py")
    assert found.examples == [] and found.skipped and "does not parse" in found.skipped[0]


def run_checker(tmp_path: Path, source: str, invariants: list[Invariant]) -> CheckResult:
    found = extract(source, "tests/t.py")
    (tmp_path / "check.py").write_text(CHECKER)
    (tmp_path / "in.json").write_text(checker_input(found.examples, invariants))
    done = subprocess.run([sys.executable, "-I", str(tmp_path / "check.py"), str(tmp_path / "in.json")],
                          capture_output=True, text=True, timeout=30)
    return read_checker_output(done.stdout + done.stderr, len(found.examples), found.skipped)


def test_an_invariant_that_raises_is_unchecked_not_a_violation(tmp_path: Path) -> None:
    result = run_checker(tmp_path, "def test_a():\n    assert wrap('a b', 1) == 'a\\nb'\n", [
        Invariant(text="lines fit", check='all(len(x) <= args[1] for x in result.split("\\n"))'),
        Invariant(text="broken rule", check="result.no_such_attribute > 0"),
        Invariant(text="for another function", fn="score", check="False"),
    ])
    assert result.violations == [] and result.checked == 1
    [unchecked] = result.unchecked
    assert unchecked.invariant == "broken rule" and "AttributeError" in unchecked.error


def test_a_violation_names_the_test_the_example_and_the_rule(tmp_path: Path) -> None:
    result = run_checker(tmp_path, "def test_a():\n    assert wrap('hi abcdefghij yo', 4) == 'hi\\nabcd\\nefgh\\nij yo'\n", [
        Invariant(text="every line is at most width characters", fn="wrap", check='all(len(x) <= args[1] for x in result.split("\\n"))'),
    ])
    [v] = result.violations
    assert v.test == "tests/t.py::test_a" and "ij yo" in v.example and v.value == "False"
    text = feedback(result.violations)
    assert "every line is at most width characters" in text and "tests/t.py::test_a" in text


def test_the_checker_sees_no_harness_builtins(tmp_path: Path) -> None:
    result = run_checker(tmp_path, "def test_a():\n    assert f(1) == 1\n", [Invariant(text="no imports", check="__import__('os')")])
    assert result.violations == [] and "NameError" in result.unchecked[0].error


def test_checker_output_that_is_not_json_is_an_error_not_a_pass() -> None:
    result = read_checker_output("Traceback: boom", 2, [])
    assert result.error and result.violations == [] and result.checked == 0
