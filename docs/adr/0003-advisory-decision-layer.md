# tev1 is advisory; human approval is the gate

The tev1 decision layer is experimental with a ~2K context, so its routing and policy answers are logged and used as a first pass but never decide a risky action. Every shell command and file write requires TUI approval even when tev1 calls it safe.

Implemented in M8 (`approval.py`): a Strands hook on `BeforeToolCallEvent` asks tev1's policy check for advice, then always awaits the TUI approval prompt. Advice only changes what the prompt shows; no code path runs a gated call on advice alone. Any failure (tev1 down or slow, the prompt raising) cancels the tool call, and `shell` and `write_file` stay gated whatever `[tools] approve` says. That is why `[tools] shell` can now be switched on in `harness.toml`.
