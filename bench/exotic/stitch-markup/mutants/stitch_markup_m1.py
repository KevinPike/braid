"""Render Stitch markup (shout, mirror and lead spans)."""

# Mutant 1: unclosed spans are emitted literally instead of raising.

from __future__ import annotations

from collections.abc import Callable

TRANSFORMS: dict[str, Callable[[str], str]] = {
    "*": str.upper,
    "~": lambda text: text[::-1],
    "^": lambda text: text[:1].upper() + text[1:],
}
ESCAPABLE = "*~^\\"


def render(src: str) -> str:
    stack: list[tuple[str, list[str]]] = [("", [])]
    i = 0
    while i < len(src):
        char = src[i]
        if char == "\\":
            if i + 1 >= len(src) or src[i + 1] not in ESCAPABLE:
                raise ValueError(f"bad escape at {i}")
            stack[-1][1].append(src[i + 1])
            i += 2
            continue
        if char in TRANSFORMS:
            if len(stack) > 1 and stack[-1][0] == char:
                marker, parts = stack.pop()
                stack[-1][1].append(TRANSFORMS[marker]("".join(parts)))
            else:
                stack.append((char, []))
        else:
            stack[-1][1].append(char)
        i += 1
    while len(stack) > 1:
        marker, parts = stack.pop()
        stack[-1][1].append(marker + "".join(parts))
    return "".join(stack[0][1])
