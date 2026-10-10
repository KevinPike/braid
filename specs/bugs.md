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
- **Status (M15, 2026-10-10):** not closed. `776a307` makes every load and unload by a Workflow run set the loading state, clears it only after a fresh `/api/ps` poll (`HarnessApp._end_loading`, also used by startup and `/model`), and writes each poll and rendered bar state to `~/.braid/status-debug.log` while a run is active. `tests/test_app_workflows.py::test_no_empty_frame_across_a_swap_...` shows no empty frame with fakes, but the original flicker was seen on a real session and has not been reproduced or re-checked there. Close this once a real swap (`/build`) leaves no `no model loaded` line in that log.
- **Planned:** M15, item 5 (specs/workflows.md § M15)

## Fixed

### Context Usage Bar not updated after workflow edits

- **Seen:** User feedback, post-M15.
- **Symptom:** After specific changes or edits (e.g., "didn't accept workflow edits"), the context usage bar display does not refresh or accurately reflect the loaded model state.
- **Likely cause (not confirmed):** The model state reporting mechanism in the UI or `Watchdog` is not robustly polling/rendering all loaded model contexts, leading to stale data being displayed.
- **Possible fix:** Ensure the state polling logic captures and displays the total memory footprint and context usage of *all* currently loaded models, including helpers, immediately following any configuration or model state change.
- **Fixed in:** `776a307` (M15). A Workflow run ends, is cancelled or fails with the chat model unloaded, the workflow model preflighted and the chat model preflighted again; `Watchdog.set_active` and `ContextPanel.set_state` make the status bar and context panel name the loaded model on the next poll. Tests: `tests/test_app_workflows.py` (`test_swap_a_completed_run_...`, `..._cancelled_...`, `..._failed_...`).

### Status bar shows only one model when several are loaded

- **Seen:** 2026-10-08, with `gemma4` and tev1 loaded together.
- **Symptom:** the status bar reports a single model, so the other loaded model's size, GPU share and keep-alive are invisible.
- **Likely cause (not confirmed):** `Watchdog.poll_once` builds `GuardState` from one `/api/ps` entry, the first that is not marked a helper (`head = next(m for m in models if m.name not in self._helpers)`). `GuardState` has no list of models, and helpers (the summarizer, tev1) are deliberately excluded from `model`, so they never reach the bar. Memory use is therefore under-reported while a helper is loaded.
- **Possible fix:** carry every loaded model in `GuardState` and render the daily driver first with the helpers after it, for example `gemma4:e4b 6.6 GiB … + tev1:0.8b 0.8 GiB`. Spill and drift alerts already cover all loaded models; only the display is affected.
- **Fixed in:** `f3dc6b3` (M15). `GuardState.models` carries every `/api/ps` entry, the active model first and helpers after; the bar renders helpers and a total that includes them. Tests: `tests/test_app_workflows.py::test_the_status_bar_renders_the_active_model_first_...`, `tests/test_status_bar.py`, `tests/guard/test_watchdog.py`.

### Commands and decisions must be inline in chat log

- **Seen:** User specification update (post-M15).
- **Symptom:** System behavior allows run commands and decision actions to appear in a separate panel or bottom view.
- **Requirement:** All run commands and subsequent decision actions must be rendered directly inline within the main chat log transcript. They must not be relegated to a separate panel or bottom view to maintain a continuous, easy-to-follow conversational flow.
- **Fixed in:** `3320d49` and `776a307` (M15). Approvals (chat shell, `write_file`, applying a Workspace diff) and Workflow interrupts are widgets mounted in the chat log that collapse to a record line; sandboxed commands a run executes appear inline. Tests: `tests/test_app_workflows.py::test_a_stuck_interrupt_and_a_chat_shell_approval_are_inline_widgets_...`, `tests/test_approval.py`.

### `search_notes` follows symlinks out of the notes folder

- **Seen:** 2026-10-08, in a pre-push audit (code reading, not reproduced).
- **Symptom:** a symlink inside the notes folder that points elsewhere (for example `notes/x.md -> ~/.ssh/config`) is read and its matching lines are returned to the model. `read_file` and `list_directory` refuse the same path.
- **Cause:** `search_notes` in `tools/builtin.py` walks `folder.rglob("*")` and reads each file directly; unlike the other tools it never passes paths through `resolve_inside`. A relative `notes_dir` is also not checked to stay inside `root`.
- **Risk:** low. The notes folder comes from `harness.toml`, not the model, and the tools only read. It is still a gap in the "bound to one project folder" promise.
- **Possible fix:** resolve the notes folder once, then skip any file whose resolved path falls outside it (`resolve_inside(folder, ...)`); add a test with a symlink pointing out of the folder, alongside the existing traversal tests in `tests/test_tools.py`.
- **Fixed in:** (uncommitted, M8). `search_notes` resolves the notes folder (a relative one must stay inside the project folder) and skips any file whose resolved path leaves it. Tests: `tests/test_tools.py`.