"""Share a depot's ration stock among a queue of requests."""

# Mutant 4: one phase: greedy requests are not held back.

from __future__ import annotations


def distribute(requests: list[tuple[str, int]], stock: int) -> dict[str, int]:
    if stock < 0:
        raise ValueError("stock is negative")
    names = [name for name, _ in requests]
    if len(set(names)) != len(names):
        raise ValueError("duplicate name")
    for name, amount in requests:
        if name in ("", "*"):
            raise ValueError("bad name")
        if amount < 1:
            raise ValueError("amount below 1")

    def greedy(name: str, amount: int) -> bool:
        return not name.startswith("*") and amount * 2 > stock

    received = dict.fromkeys(names, 0)
    left = stock
    phases = (requests,)
    for phase in phases:
        while left > 0 and any(received[name] < amount for name, amount in phase):
            for name, amount in phase:
                share = 3 if name.startswith("*") else 2
                give = min(share, amount - received[name], left)
                received[name] += give
                left -= give
    return received
