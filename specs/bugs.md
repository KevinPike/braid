# Bugs and rough edges

Tabled issues that are not on a milestone's critical path. Newest first. Close an entry by moving it to **Fixed** with the commit.

## Open

### Status bar shows only one model when several are loaded

- **Seen:** 2026-10-08, with `gemma4` and tev1 loaded together.
- **Symptom:** the status bar reports a single model, so the other loaded model's size, GPU share and keep-alive are invisible.
- **Likely cause (not confirmed):** `Watchdog.poll_once` builds `GuardState` from one `/api/ps` entry, the first that is not marked a helper (`head = next(m for m in models if m.name not in self._helpers)`). `GuardState` has no list of models, and helpers (the summarizer, tev1) are deliberately excluded from `model`, so they never reach the bar. Memory use is therefore under-reported while a helper is loaded.
- **Possible fix:** carry every loaded model in `GuardState` and render the daily driver first with the helpers after it, for example `gemma4:e4b 6.6 GiB … + tev1:0.8b 0.8 GiB`. Spill and drift alerts already cover all loaded models; only the display is affected.

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
