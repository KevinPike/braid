"""Render Stitch markup (shout, mirror and lead spans)."""

# Mutant 4: unknown escapes pass the character through.

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
            if i + 1 >= len(src):
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
    if len(stack) > 1:
        raise ValueError("unclosed span")
    return "".join(stack[0][1])
