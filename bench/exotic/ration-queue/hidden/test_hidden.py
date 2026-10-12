# braid-hidden-canary: ration-queue-f2657e5d989650ad
"""Hidden tests for ration-queue. Never show this file to the model under test."""

import pytest
from ration_queue import distribute


# GOAL examples
def test_goal_examples():
    assert distribute([("ana", 3), ("*bo", 4)], 10) == {"ana": 3, "*bo": 4}
    assert distribute([("ana", 9), ("cy", 4), ("*bo", 5)], 10) == {"ana": 1, "cy": 4, "*bo": 5}


# Rules 4 and 5: rounds of 2 for commoners, capped by need and by stock left mid-round
@pytest.mark.parametrize(("requests", "stock", "expected"), [
    ([], 5, {}),
    ([("ana", 2)], 5, {"ana": 2}),
    ([("ana", 5)], 10, {"ana": 5}),
    ([("ana", 3), ("bo", 1), ("*cy", 7)], 40, {"ana": 3, "bo": 1, "*cy": 7}),
    ([("a1", 4), ("a2", 4), ("a3", 4)], 9, {"a1": 4, "a2": 3, "a3": 2}),
    ([("ana", 1), ("bo", 1), ("cy", 1)], 2, {"ana": 1, "bo": 1, "cy": 0}),
    ([("ana", 3)], 6, {"ana": 3}),
    ([("x", 1)], 0, {"x": 0}),
])
def test_rounds(requests, stock, expected):
    assert distribute(requests, stock) == expected


# Rules 1 and 4: elders get 3 per round
@pytest.mark.parametrize(("requests", "stock", "expected"), [
    ([("*el", 5), ("bo", 2), ("cy", 2)], 5, {"*el": 3, "bo": 2, "cy": 0}),
    ([("*el", 7)], 4, {"*el": 4}),
    ([("**x", 4)], 3, {"**x": 3}),
    ([("*a", 3), ("*b", 3)], 4, {"*a": 3, "*b": 1}),
    ([("a", 2), ("*b", 6), ("c", 2)], 7, {"a": 2, "*b": 3, "c": 2}),
])
def test_elder_share(requests, stock, expected):
    assert distribute(requests, stock) == expected


# Rules 2 and 3: greedy commoners (amount x 2 > stock) wait for the second phase
@pytest.mark.parametrize(("requests", "stock", "expected"), [
    ([("ana", 6)], 10, {"ana": 6}),
    ([("ana", 6), ("bo", 5)], 10, {"ana": 5, "bo": 5}),
    ([("ana", 4), ("bo", 3), ("cy", 3)], 8, {"ana": 4, "bo": 2, "cy": 2}),
    ([("g1", 8), ("x", 1), ("g2", 9)], 10, {"g1": 5, "x": 1, "g2": 4}),
    ([("ana", 1), ("*bo", 2)], 0, {"ana": 0, "*bo": 0}),
    ([("big", 50), ("s1", 3), ("s2", 4)], 20, {"big": 13, "s1": 3, "s2": 4}),
    ([("g1", 7), ("g2", 7)], 6, {"g1": 4, "g2": 2}),
    ([("ana", 3)], 5, {"ana": 3}),
    ([("*el", 2), ("bo", 9)], 9, {"*el": 2, "bo": 7}),
])
def test_greedy_phase(requests, stock, expected):
    assert distribute(requests, stock) == expected


# Rule 2: elders are never greedy, whatever their amount
def test_elders_never_greedy():
    assert distribute([("*el", 7), ("cy", 4)], 6) == {"*el": 6, "cy": 0}


# Rule 6: errors
@pytest.mark.parametrize(("requests", "stock"), [
    ([("ana", 2)], -1), ([("ana", 0)], 5), ([("ana", -2)], 5), ([("", 2)], 5), ([("*", 2)], 5),
    ([("ana", 1), ("ana", 2)], 5), ([("*el", 1), ("bo", 1), ("*el", 1)], 5),
    ([("bo", 1), ("*", 3)], 5),
])
def test_errors(requests, stock):
    with pytest.raises(ValueError):
        distribute(requests, stock)
