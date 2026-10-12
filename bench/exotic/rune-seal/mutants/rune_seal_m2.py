"""Seal and check rune codes with a weighted check character."""

# Mutant 2: no echo rule.

from __future__ import annotations

ALPHABET = "0123456789ACDEFHJKLMNPRTVWXY"
WEIGHTS = (1, 3, 7)
LOWER = "abcdefghijklmnopqrstuvwxyz"


def _normal(text: str) -> str:
    return "".join(c.upper() if c in LOWER else c for c in text if c not in "- ")


def _check(normal: str) -> str:
    total = len(normal)
    for i, c in enumerate(normal):
        total += ALPHABET.index(c) * WEIGHTS[i % 3]
    return ALPHABET[total % 28]


def seal(code: str) -> str:
    normal = _normal(code)
    if not normal:
        raise ValueError("empty code")
    if any(c not in ALPHABET for c in normal):
        raise ValueError("character outside the rune alphabet")
    return f"{normal}-{_check(normal)}"


def is_sealed(text: str) -> bool:
    normal = _normal(text)
    if len(normal) < 2 or any(c not in ALPHABET for c in normal):
        return False
    return _check(normal[:-1]) == normal[-1]
