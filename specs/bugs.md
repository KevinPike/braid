# Bugs and rough edges

Tabled issues that are not on a milestone's critical path. Newest first. Close an entry by moving it to **Fixed** with the commit.

## Open

### `search_notes` follows symlinks out of the notes folder

- **Seen:** 2026-10-08, in a pre-push audit (code reading, not reproduced).
- **Symptom:** a symlink inside the notes folder that points elsewhere (for example `notes/x.md -> ~/.ssh/config`) is read and its matching lines are returned to the model. `read_file` and `list_directory` refuse the same path.
- **Cause:** `search_notes` in `tools/builtin.py` walks `folder.rglob("*")` and reads each file directly; unlike the other tools it never passes paths through `resolve_inside`. A relative `notes_dir` is also not checked to stay inside `root`.
- **Risk:** low. The notes folder comes from `harness.toml`, not the model, and the tools only read. It is still a gap in the "bound to one project folder" promise.
- **Possible fix:** resolve the notes folder once, then skip any file whose resolved path falls outside it (`resolve_inside(folder, ...)`); add a test with a symlink pointing out of the folder, alongside the existing traversal tests in `tests/test_tools.py`.

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
