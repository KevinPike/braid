# braid-hidden-canary: ferry-fare-8b3c5dae37bae81b
"""Hidden tests for ferry-fare. Never show this file to the model under test."""

import pytest
from ferry_fare import fare


# GOAL examples
def test_goal_examples():
    assert fare([30, 8], "car") == 22
    assert fare([70, 40, 10, 2], "car") == 20


# Rule 1: passenger fares by age band, and their boundaries
@pytest.mark.parametrize(("ages", "expected"), [
    ([0], 0), ([3], 0), ([4], 3), ([15], 3), ([16], 7), ([64], 7), ([65], 4), ([120], 4),
    ([], 0), ([2, 9, 33], 10), ([70, 3], 4),
])
def test_passenger_fare_by_age(ages, expected):
    assert fare(ages, "none") == expected


# Rule 2: vehicle fees
@pytest.mark.parametrize(("ages", "vehicle", "expected"), [
    ([40], "bike", 9), ([40], "car", 19), ([40], "van", 25), ([12], "bike", 5), ([16], "car", 19),
    ([16, 2], "van", 25),
])
def test_vehicle_fee(ages, vehicle, expected):
    assert fare(ages, vehicle) == expected


# Rule 3: group waiver counts only passengers with a fare above 0, and waives exactly one lowest fare
@pytest.mark.parametrize(("ages", "expected"), [
    ([16, 16, 16, 16], 21), ([20, 5, 70, 30], 18), ([20, 20, 4, 4, 4], 20), ([1, 2, 30, 30, 30], 21),
    ([1, 30, 30, 30, 9], 21), ([70, 70, 70, 70], 12), ([30, 30, 30, 30, 30, 30], 35), ([3, 4, 15, 16, 65], 14),
])
def test_group_waiver(ages, expected):
    assert fare(ages, "none") == expected


# Rule 4: carpool halves a car's fee for 3+ passengers of any age; never a van's or a bike's
@pytest.mark.parametrize(("ages", "vehicle", "expected"), [
    ([30, 30], "car", 26), ([30, 30, 30], "car", 27), ([30, 1, 2], "car", 13), ([30, 30, 30], "van", 39), ([25, 30, 40, 50], "van", 39),
    ([30, 30, 30], "bike", 23), ([30, 30, 30, 30], "car", 27), ([18, 5], "car", 22),
])
def test_carpool(ages, vehicle, expected):
    assert fare(ages, vehicle) == expected


# Rules 3 and 4 together, then rule 5's sum
def test_waiver_and_carpool_together():
    assert fare([40, 40, 10, 10, 1], "car") == 23


# Rule 6: errors
@pytest.mark.parametrize(("ages", "vehicle"), [
    ([-1], "none"), ([121], "none"), ([30, -2], "car"), ([30], "Car"), ([30], "boat"), ([30], ""),
    ([15], "car"), ([10, 2], "van"), ([], "car"), ([], "van"), ([], "bike"),
])
def test_errors(ages, vehicle):
    with pytest.raises(ValueError):
        fare(ages, vehicle)
