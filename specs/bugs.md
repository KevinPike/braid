# Bugs and rough edges

Tabled issues that are not on a milestone's critical path. Newest first. Close an entry by moving it to **Fixed** with the commit.

## Open

### Status bar still flickers when the model reloads

- **Seen:** 2026-10-08, after the M3 work, on a real session.
- **Symptom:** when the model is (re)loaded the status bar briefly shows the model, then "no model loaded", then the model again; the load as a whole feels erratic.
- **Already tried:** Preflight no longer unloads a model that is already loaded at the pinned `num_ctx` (`788d623`), and the bar shows `loading <model>…` during startup and `/model` switches. Both are covered by fakes only, not verified live.
- **Suspects (unconfirmed):**
  - Reloads outside Preflight (eviction, `num_ctx` drift, the summarizer's load and unload) never set the loading state, so the 2 s `/api/ps` poll shows the gap.
  - Preflight still unloads and warms when the loaded `num_ctx` differs, and the poller publishes the empty state in between.
  - `Watchdog.poll_once` takes the first non-helper entry in `/api/ps`; a stale entry from before the harness started can be shown first.
- **Next step:** reproduce with a log of each `/api/ps` poll and each state the status bar renders, to see which of the above produces the empty frame.

## Fixed

_None yet._
