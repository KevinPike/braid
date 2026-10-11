# Exotic katas

Six invented katas whose rules exist nowhere else, so a model cannot have memorized them. Each one has a held-out
hidden test suite, which lets a run be graded two ways: Green on the model's own tests, and correct on tests it never
saw.

| kata | difficulty | function |
|---|---|---|
| ferry-fare | easy | `fare(ages, vehicle)`: age-banded fares, group waiver, car carpool |
| velm-calendar | easy | `velm_date(day)`: 13x28-day calendar with a Hollow day in long years |
| rune-seal | medium | `seal(code)`, `is_sealed(text)`: weighted mod-28 check character with an echo rule |
| orchard-score | medium | `score(grid)`: board scoring with crowding, wasp spoiling and row bonuses |
| ration-queue | hard | `distribute(requests, stock)`: round-robin shares, elders, a deferred greedy phase |
| stitch-markup | hard | `render(src)`: nested shout/mirror/lead spans, escapes, unclosed-span errors |

Each `<kata>/` holds:

- `GOAL.md` and `summary.txt`: the task as a model sees it (same shape as `examples/graphs/make_katas.py`).
- `plan.json`: a Plan with Behaviors and Invariants. Its Behavior values differ from the hidden ones.
- `reference/<module>.py`: a reference implementation.
- `hidden/test_hidden.py`: the held-out suite, 30 to 60 cases.
- `mutants/<module>_m1.py` ... `_m5.py`: plausible buggy variants. Each one must fail the hidden suite.

## The hidden tests never enter a model's context

A hidden suite measures correctness only while the model has never seen it. If a test file, a reference, or a mutant
reaches a prompt, the model can fit those exact values and the grade becomes "Green on its own tests" again. So:

- Give the model `GOAL.md` (and `plan.json` if needed), nothing else from this directory.
- Grade a run by copying its module and `hidden/test_hidden.py` into a scratch directory outside the kata repo, after
  the Workflow ends.
- Each hidden file starts with a `# braid-hidden-canary: <kata>-<hex>` line. Assert that no model prompt contains
  that canary.

## Verify

```
uv run python bench/exotic/verify.py
```

For every kata, verify.py checks that:

- the reference passes the hidden suite;
- every mutant fails it;
- each Plan Invariant holds on every example the F3 extractor finds;
- `plan.json` loads as a Plan, and no Behavior quotes a hidden call;
- each canary is unique and appears in no other file.

It runs pytest in temporary directories and exits non-zero on any failure.
