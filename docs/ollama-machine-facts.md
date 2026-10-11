# Ollama machine facts (frozen path, ADR 0007)

True as of 2026-10-10. They matter only when working on `guard/`, Calibration, Preflight or an Ollama-backed run.

- 24 GB Mac; `sysctl iogpu.wired_limit_mb` is 0 (default), so the Metal limit is unmeasured until M1's preflight measures it.
- Installed models: `gemma4:e4b`, `gemma4:e2b`, `gemma4:12b`, `gemma4:12b-mlx`, `tev1:0.8b`. The 26B MoE, `gpt-oss:20b` and `nomic-embed-text` from the plan are not pulled yet. `gemma4:12b` is too memory-heavy for workflow runs here (swap past 5 GiB, 3–5 tok/s); live checks use `gemma4:e4b`.
- `gemma4:e4b` has sliding-window and shared-KV layers (`/api/show`), so the KV formula overestimates it; calibrate (ADR 0002).
- macOS 27: `sandbox-exec` works (verified 2026-10-09); no container runtime is installed. `llama3.1:8b` is not pulled.
- Ollama 0.35.1; `/api/ps` entries include `context_length`.
- `/api/ps` `size`/`size_vram` are wrong for models with a speculative draft (all `gemma4` here; ollama#17251), so never trust them alone for budgets or spill detection (ADR 0002).
