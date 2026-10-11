"""Score a finished orchard in the board game Bramble."""

# Mutant 5: crowded pear scores 2.

from __future__ import annotations


def score(grid: list[str]) -> int:
    if not grid or any(not row for row in grid):
        raise ValueError("the grid needs rows with tiles")
    if len({len(row) for row in grid}) != 1:
        raise ValueError("rows differ in length")
    if any(c not in "APW." for row in grid for c in row):
        raise ValueError("unknown tile")
    height, width = len(grid), len(grid[0])

    def neighbours(r: int, c: int) -> list[str]:
        cells = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        return [grid[y][x] for y, x in cells if 0 <= y < height and 0 <= x < width]

    total = 0
    for r, row in enumerate(grid):
        for c, tile in enumerate(row):
            if tile not in "AP" or "W" in neighbours(r, c):
                continue
            if tile == "A":
                total += 1
            else:
                total += 2 if "P" in neighbours(r, c) else 3
        if "." not in row and "W" not in row:
            total += 2
    return total
