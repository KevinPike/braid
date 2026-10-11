# braid-hidden-canary: orchard-score-0669533b9ffc8b14
"""Hidden tests for orchard-score. Never show this file to the model under test."""

import pytest
from orchard_score import score


# GOAL examples
def test_goal_examples():
    assert score(["AP.", "PW."]) == 1
    assert score(["APA", "PPA"]) == 10


# Rules 1, 4 and 5: single tiles and the row bonus
@pytest.mark.parametrize(("grid", "expected"), [
    (["A"], 3), (["."], 0), (["P"], 5), (["W"], 0), (["...", "..."], 0), (["AAA"], 5), (["WW"], 0),
])
def test_single_tiles(grid, expected):
    assert score(grid) == expected


# Rule 2: crowded pears, orthogonal neighbours only
@pytest.mark.parametrize(("grid", "expected"), [
    (["PP"], 4), (["P.P"], 6), (["P", "P"], 6), (["P.", ".P"], 6), (["PPP"], 5), (["PA", "AP"], 12), (["PP", "PP"], 8),
    (["P.P", ".P."], 9),
])
def test_crowded_pears(grid, expected):
    assert score(grid) == expected


# Rule 3: wasps spoil orthogonal neighbours; spoiled pears still crowd
@pytest.mark.parametrize(("grid", "expected"), [
    (["AW"], 0), (["A.", ".W"], 1), (["PWP"], 0), (["PPW"], 1), (["W.", "PP"], 3), (["A.A", "WWW"], 0), (["W", "A"], 2),
    (["P", "W", "P"], 4), (["P", "P", "W"], 5), (["APW", "PAP"], 7),
])
def test_wasps(grid, expected):
    assert score(grid) == expected


# Rules 3 and 4 together: a row bonus survives fruit spoiled from another row
@pytest.mark.parametrize(("grid", "expected"), [
    (["WA", "AA"], 3), (["AAA", "AWA", "AAA"], 8),
])
def test_bonus_with_spoiled_fruit(grid, expected):
    assert score(grid) == expected


# Rule 6: errors
@pytest.mark.parametrize("grid", [[], [""], ["AA", "A"], ["A", ""], ["AB"], ["a"], ["A P"], ["A", "AA"], ["W", "x"]])
def test_errors(grid):
    with pytest.raises(ValueError):
        score(grid)
