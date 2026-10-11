"""F3 prototype: would Plan Invariants have caught the four tests Haiku disputed in the F1 live check?

    uv run python examples/graphs/invariants_proto.py

The tests are copied from the F1 run logs (~/.braid/runs/20261010-14*-claude-haiku-5-5.jsonl); the Invariants are the
ones added by hand to bench/plans/disputed/. No model runs: the extractor and the checker are deterministic. The checker
runs as a subprocess here; the Execution loop runs it in the Sandbox.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from harness.workflow.invariants import CHECKER, Invariant, checker_input, extract, read_checker_output

WRAP_WIDTH = Invariant(
    text="every line is at most width characters", fn="wrap",
    check='all(len(line) <= args[1] for line in result.split("\\n"))')
WRAP_GREEDY = Invariant(
    text="greedy: no line has room for the first word of the line after it", fn="wrap",
    check='all(len(a) + 1 + len(b.split(" ")[0]) > args[1] for a, b in zip(result.split("\\n"), result.split("\\n")[1:]))')
BOWL_PINS = Invariant(
    text="bonuses only add: a game scores at least the pins knocked down", fn="score", check="result >= sum(args[0])")
BOWL_FIRST_STRIKE = Invariant(
    text="a strike in the first frame adds the next two rolls on top of the pins", fn="score",
    check="args[0][0] != 10 or result >= sum(args[0]) + args[0][1] + args[0][2]")
BOWL_RANGE = Invariant(text="a game scores 0 to 300", fn="score", check="0 <= result <= 300")

CASES: list[tuple[str, str, list[Invariant]]] = [
    ("word-wrap, Claude Plan, Behavior 5", '''
from word_wrap import wrap


def test_leftover_piece_of_split_word_starts_line_that_following_words_join():
    assert wrap("hi abcdefghij yo", 4) == "hi\\nabcd\\nefgh\\nij yo"
''', [WRAP_WIDTH, WRAP_GREEDY]),
    ("bowling, local Plan, Behavior 2", '''
from bowling import score


def test_score_of_standard_open_frame_is_sum_of_two_rolls():
    assert score([3, 4] + [0] * 18) == 7


def test_open_frames_sum_their_rolls_across_the_game():
    assert score([1, 2, 3, 4, 5, 3, 0, 9, 2, 6, 4, 4, 7, 1, 0, 8, 1, 2, 3, 5]) == 3 + 7 + 8 + 3 + 9 + 8 + 8 + 1 + 3 + 8
''', [BOWL_PINS, BOWL_FIRST_STRIKE, BOWL_RANGE]),
    ("bowling, local Plan, Behavior 3", '''
from bowling import score


def test_strike_scores_ten_plus_next_two_rolls_in_first_frame():
    rolls = [10, 3, 4] + [0] * 16
    assert score(rolls) == 17


def test_strike_scores_ten_plus_next_two_rolls_in_middle_frame():
    rolls = [0, 0] * 4 + [10, 5, 2] + [0] * 8
    assert score(rolls) == 17
''', [BOWL_PINS, BOWL_FIRST_STRIKE, BOWL_RANGE]),
    ("word-wrap, local Plan, Behavior 2", '''
import pytest

from word_wrap import wrap


@pytest.mark.parametrize(
    "text, width, expected",
    [
        ("the quick brown fox", 10, "the quick\\nbrown fox"),
        ("the quick brown fox", 19, "the quick brown fox"),
        ("the quick brown fox", 9, "the quick\\nbrown\\nfox"),
        ("aa bb cc dd", 5, "aa bb\\ncc dd"),
    ],
)
def test_wrap_fits_as_many_whole_words_per_line(text, width, expected):
    assert wrap(text, width) == expected
''', [WRAP_WIDTH, WRAP_GREEDY]),
]


def main() -> None:
    caught = 0
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(CHECKER)
        for title, source, invariants in CASES:
            found = extract(source, "tests/test_behavior.py")
            data = Path(tmp) / "input.json"
            data.write_text(checker_input(found.examples, invariants))
            done = subprocess.run([sys.executable, "-I", str(script), str(data)], capture_output=True, text=True, timeout=30)
            result = read_checker_output(done.stdout + done.stderr, len(found.examples), found.skipped)
            print(f"\n== {title}: {len(found.examples)} examples, {len(found.skipped)} skipped, {result.checked} checks")
            for v in result.violations:
                print(f"   VIOLATION {v.test.split('::')[1]}: {v.example}\n             breaks: {v.invariant}")
            for u in result.unchecked:
                print(f"   unchecked {u.test}: {u.invariant}: {u.error}")
            for s in found.skipped:
                print(f"   skipped {s}")
            caught += bool(result.violations)
    print(f"\n{caught}/{len(CASES)} disputed Behaviors have a test the Invariants reject")


if __name__ == "__main__":
    main()
