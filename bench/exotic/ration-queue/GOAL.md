# ration-queue

`distribute(requests: list[tuple[str, int]], stock: int) -> dict[str, int]` shares a depot's `stock` of ration
units among `requests`, given in queue order as `(name, amount wanted)` pairs, and returns how many units each name
receives. Every requested name is a key of the result, with 0 when it receives nothing. Rules:

1. A name starting with `*` is an elder; every other name is a commoner.
2. A commoner whose amount is more than half the starting stock (amount × 2 > stock) is greedy. Elders are never
   greedy, whatever their amount.
3. Units are handed out in two phases: first to all requests that are not greedy, then, with whatever stock is left,
   to the greedy ones. Each phase hands out units in rounds.
4. In a round, every request of the phase that has not yet received its full amount, taken in queue order, receives
   its share: 3 units for an elder, 2 for a commoner, but never more than it still needs and never more than the
   stock left at that moment (which may be 0).
5. A phase ends when every request in it has received its full amount, or when the stock left is 0.
6. `ValueError` is raised when `stock` is negative, an amount is below 1, a name is the empty string or exactly
   `"*"`, or two requests have the same name (`"bo"` and `"*bo"` are different names). Arguments always have the
   annotated types; nothing else needs checking.

- `distribute([("ana", 3), ("*bo", 4)], 10)` returns `{"ana": 3, "*bo": 4}`
- `distribute([("ana", 9), ("cy", 4), ("*bo", 5)], 10)` returns `{"ana": 1, "cy": 4, "*bo": 5}`
