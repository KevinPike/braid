# velm-calendar

`velm_date(day: int) -> str` converts a day count to a date in the Velm calendar. Day 1 is the first day of month 1
of year 1, and days run on with no gaps. Rules:

1. Every year has 13 months of 28 days each, numbered from 1; that is 364 days.
2. A long year has one extra day, Hollow, which comes after day 28 of month 13 (so a long year has 365 days).
3. A year is long when its number is divisible by 5, except that a year divisible by 30 is not long unless it is
   also divisible by 150. So years 5, 10, 25 and 150 are long; years 30, 60 and 120 are not; years 1 to 4 are not.
4. An ordinary date is written `"Y-MM-DD"`: the year as a plain integer with no padding, then the month and the day
   as two digits each (for example `"7-03-09"`). Hollow is written `"Y-HOLLOW"` (for example `"10-HOLLOW"`).
5. `day` below 1 raises `ValueError`. `day` is always an `int`; nothing else needs checking.

- `velm_date(365)` returns `"2-01-01"`
- `velm_date(1821)` returns `"5-HOLLOW"`
