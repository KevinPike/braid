# braid-hidden-canary: stitch-markup-b90d8e1a2f5e7eb6
"""Hidden tests for stitch-markup. Never show this file to the model under test."""

import pytest
from stitch_markup import render


# GOAL examples
def test_goal_examples():
    assert render("say *hi* ~olleh~") == "say HI hello"
    assert render("*a~bc~*") == "ACB"


# Rule 4 and the three transformations
@pytest.mark.parametrize(("src", "expected"), [
    ("", ""), ("plain text", "plain text"), ("*hi*", "HI"), ("~abc~", "cba"), ("^word^", "Word"),
    ("^aBC^", "ABC"), ("^hELLO world^", "HELLO world"), ("x ^1st^ y", "x 1st y"),
])
def test_transformations(src, expected):
    assert render(src) == expected


# Rule 2: empty spans render as nothing
@pytest.mark.parametrize(("src", "expected"), [("**", ""), ("^^", ""), ("~~x", "x"), ("a**b", "ab"), ("*~~*", "")])
def test_empty_spans(src, expected):
    assert render(src) == expected


# Rules 1 and 2: nesting, inner content rendered first, same marker closes the innermost span
@pytest.mark.parametrize(("src", "expected"), [
    ("~a*bc*~", "CBa"), ("~^ab^~", "bA"), ("^~ab~^", "Ba"), ("~ab*cd*~", "DCba"), ("*a*b*c*", "AbC"),
    ("~a~~b~", "ab"), ("*a^b^c*", "ABC"), ("~x^ab^y~", "ybAx"),
])
def test_nesting(src, expected):
    assert render(src) == expected


# Rule 3: escapes are literal ordinary text, transformed by enclosing spans
@pytest.mark.parametrize(("src", "expected"), [
    ("\\*not\\*", "*not*"), ("*a\\*b*", "A*B"), ("~a\\~b~", "b~a"), ("\\\\", "\\"), ("*\\^x*", "^X"),
    ("^\\*a^", "*a"), ("a\\\\*b*", "a\\B"),
])
def test_escapes(src, expected):
    assert render(src) == expected


# Rule 5: errors (bad escapes, unclosed or crossing spans)
@pytest.mark.parametrize("src", ["*abc", "~a", "*a~b*", "^x*y^", "a\\", "\\n", "\\a", "*a^b*^", "~~~", "*\\x*"])
def test_errors(src):
    with pytest.raises(ValueError):
        render(src)
