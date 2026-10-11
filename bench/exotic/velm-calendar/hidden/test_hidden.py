# braid-hidden-canary: velm-calendar-6405b61a0aba0a67
"""Hidden tests for velm-calendar. Never show this file to the model under test."""

import pytest
from velm_calendar import velm_date


# GOAL examples
def test_goal_examples():
    assert velm_date(365) == "2-01-01"
    assert velm_date(1821) == "5-HOLLOW"


# Rules 1 and 4: 28-day months, 13 per year, zero-padded month and day
@pytest.mark.parametrize(("day", "expected"), [
    (1, "1-01-01"), (2, "1-01-02"), (336, "1-12-28"), (337, "1-13-01"), (28, "1-01-28"), (29, "1-02-01"), (57, "1-03-01"), (100, "1-04-16"), (364, "1-13-28"),
    (1456, "4-13-28"), (1457, "5-01-01"), (1521, "5-03-09"),
])
def test_ordinary_dates(day, expected):
    assert velm_date(day) == expected


# Rules 2 and 3: Hollow ends a long year (divisible by 5)
@pytest.mark.parametrize(("day", "expected"), [
    (1820, "5-13-28"), (1822, "6-01-01"), (3641, "10-13-28"), (3642, "10-HOLLOW"), (3643, "11-01-01"), (9105, "25-HOLLOW"),
    (9106, "26-01-01"),
])
def test_long_years(day, expected):
    assert velm_date(day) == expected


# Rule 3 exceptions: divisible by 30 is not long, unless divisible by 150
@pytest.mark.parametrize(("day", "expected"), [
    (10925, "30-13-28"), (10926, "31-01-01"), (21850, "60-13-28"), (21851, "61-01-01"), (43700, "120-13-28"),
    (43701, "121-01-01"), (54625, "150-13-28"), (54626, "150-HOLLOW"), (54627, "151-01-01"), (109252, "300-HOLLOW"),
    (109253, "301-01-01"),
])
def test_long_year_exceptions(day, expected):
    assert velm_date(day) == expected


# Rule 5: errors
@pytest.mark.parametrize("day", [0, -1, -364])
def test_errors(day):
    with pytest.raises(ValueError):
        velm_date(day)
