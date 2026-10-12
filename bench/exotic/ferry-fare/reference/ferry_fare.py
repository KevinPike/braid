"""Price one party's crossing on the Skelling ferry."""

from __future__ import annotations

VEHICLE_FEES = {"none": 0, "bike": 2, "car": 12, "van": 18}


def _passenger_fare(age: int) -> int:
    if age <= 3:
        return 0
    if age <= 15:
        return 3
    if age <= 64:
        return 7
    return 4


def fare(ages: list[int], vehicle: str) -> int:
    if vehicle not in VEHICLE_FEES:
        raise ValueError(f"unknown vehicle: {vehicle!r}")
    if any(age < 0 or age > 120 for age in ages):
        raise ValueError("age out of range")
    if vehicle in ("car", "van") and not any(age >= 16 for age in ages):
        raise ValueError(f"a {vehicle} needs a passenger aged 16 or over")
    if vehicle == "bike" and not ages:
        raise ValueError("a bike needs a passenger")
    paying = [p for p in map(_passenger_fare, ages) if p > 0]
    total = sum(paying)
    if len(paying) >= 4:
        total -= min(paying)
    fee = VEHICLE_FEES[vehicle]
    if vehicle == "car" and len(ages) >= 3:
        fee = 6
    return total + fee
