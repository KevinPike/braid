"""Seal and check rune codes with a weighted check character."""

# Mutant 3: echoed characters do not advance the weight cycle.

from __future__ import annotations

ALPHABET = "0123456789ACDEFHJKLMNPRTVWXY"
WEIGHTS = (1, 3, 7)
LOWER = "abcdefghijklmnopqrstuvwxyz"


def _normal(text: str) -> str:
    return "".join(c.upper() if c in LOWER else c for c in text if c not in "- ")


def _check(normal: str) -> str:
    total = len(normal)
    k = 0
    for i, c in enumerate(normal):
        if i > 0 and normal[i - 1] == c:
            continue
        total += ALPHABET.index(c) * WEIGHTS[k % 3]
        k += 1
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
