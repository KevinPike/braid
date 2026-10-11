"""Convert day counts to dates in the Velm calendar."""

# Mutant 1: every fifth year is long, no exceptions.

from __future__ import annotations


def _is_long(year: int) -> bool:
    return year % 5 == 0


def velm_date(day: int) -> str:
    if day < 1:
        raise ValueError("day must be 1 or more")
    year, rest = 1, day
    while rest > (length := 365 if _is_long(year) else 364):
        rest -= length
        year += 1
    if rest == 365:
        return f"{year}-HOLLOW"
    month, day_of_month = divmod(rest - 1, 28)
    return f"{year}-{month + 1:02d}-{day_of_month + 1:02d}"
