"""Convert day counts to dates in the Velm calendar."""

# Mutant 3: Hollow is the first day of a long year.

from __future__ import annotations


def _is_long(year: int) -> bool:
    return year % 5 == 0 and (year % 30 != 0 or year % 150 == 0)


def velm_date(day: int) -> str:
    if day < 1:
        raise ValueError("day must be 1 or more")
    year, rest = 1, day
    while rest > (length := 365 if _is_long(year) else 364):
        rest -= length
        year += 1
    if _is_long(year):
        if rest == 1:
            return f"{year}-HOLLOW"
        rest -= 1
    month, day_of_month = divmod(rest - 1, 28)
    return f"{year}-{month + 1:02d}-{day_of_month + 1:02d}"
