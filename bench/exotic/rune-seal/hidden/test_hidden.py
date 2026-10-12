# braid-hidden-canary: rune-seal-b88ed918d58ff7b5
"""Hidden tests for rune-seal. Never show this file to the model under test."""

import pytest
from rune_seal import is_sealed, seal


# GOAL examples
def test_goal_examples():
    assert seal("ac 77") == "AC77-D"
    assert is_sealed("ac77d") is True


# Rules 2 and 4: weights 1, 3, 7 from the left, plus the length, modulo 28
@pytest.mark.parametrize(("code", "expected"), [
    ("0", "0-1"), ("1", "1-2"), ("A", "A-C"), ("Y", "Y-0"), ("12", "12-9"), ("21", "21-7"), ("123", "123-3"),
    ("1234", "1234-8"), ("XY", "XY-W"),
])
def test_weights_and_check(code, expected):
    assert seal(code) == expected


# Rule 3: echo rule, judged on the normal form, keeping the weight cycle
@pytest.mark.parametrize(("code", "expected"), [
    ("AA", "AA-D"), ("AAA", "AAA-E"), ("7777", "7777-C"), ("7A7A", "7A7A-J"), ("DDEE", "DDEE-T"),
])
def test_echo(code, expected):
    assert seal(code) == expected


# Rule 1: normal form (lowercase up, hyphens and spaces dropped)
@pytest.mark.parametrize(("code", "expected"), [
    ("aa", "AA-D"), ("a-a", "AA-D"), ("a a", "AA-D"), ("7 7 7 7", "7777-C"), ("dd-ee", "DDEE-T"), ("-x-y-", "XY-W"),
])
def test_normal_form(code, expected):
    assert seal(code) == expected


# Rule 5: errors
@pytest.mark.parametrize("code", ["", "-", " - ", "B", "ao", "AC!", "A_C", "A\tC", "ai", "A.C"])
def test_seal_errors(code):
    with pytest.raises(ValueError):
        seal(code)


# is_sealed: normal form, length at least 2, alphabet, check character; never raises
@pytest.mark.parametrize(("text", "expected"), [
    ("AC77-D", True), ("AC77-C", False), ("1-2", True), ("12-9", True), ("12 9", True), ("21-9", False),
    ("y0", True), ("ddee-t", True), ("DEDE-T", False), ("D", False), ("", False), ("--", False),
    ("AB-1", False), ("AC77-D!", False),
])
def test_is_sealed(text, expected):
    assert is_sealed(text) == expected
