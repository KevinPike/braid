"""Loads harness.toml: the model profile and paths."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ThinkingDisplay = Literal["summarized", "omitted"]


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"


Backend = Literal["anthropic", "ollama"]
API_PREFIX = "claude-"  # a model id with this prefix runs on the Anthropic API, whatever ``backend`` says


class ProfileConfig(BaseModel):
    """The chat agent. ``backend = "anthropic"`` (the default, ADR 0007) runs ``anthropic_model`` on the API with no Guard;
    ``"ollama"`` runs ``model`` through the guard client (the frozen path). A ``claude-`` ``model`` picks the API too."""

    backend: Backend = "anthropic"
    anthropic_model: str = "claude-haiku-5-5"
    effort: Effort | None = None  # Anthropic only; None leaves the model's default
    max_tokens: int = 16000  # Anthropic only: cap on generated tokens per call
    context_window: int | None = None  # Anthropic only: compaction's window; default the model's real one
    model: str = "gemma4:e4b"
    num_ctx: int = 32768
    keep_alive: str = "30m"
    system_prompt: str = "You are a concise, helpful assistant."
    # Tried in order when `model` does not fit the GPU budget (preflight step-down).
    fallbacks: list[str] = []

    @property
    def chat_backend(self) -> Backend:
        return "anthropic" if self.model.startswith(API_PREFIX) else self.backend

    @property
    def chat_model(self) -> str:
        """The model the chat agent starts on: ``model`` when it is a ``claude-`` id or the backend is Ollama."""
        if self.model.startswith(API_PREFIX) or self.backend == "ollama":
            return self.model
        return self.anthropic_model


class ContextConfig(BaseModel):
    """Compaction (M3). ``summarize`` is the custom strategy; ``sliding_window`` is Strands' built-in."""

    strategy: Literal["none", "sliding_window", "summarize"] = "summarize"
    window_size: int = 40  # sliding_window: messages kept
    summarizer_model: str = "gemma4:e2b"  # falls back to the daily driver when not pulled
    summarizer_num_ctx: int = 16384
    ceiling: float = 0.9  # no call may pass this share of num_ctx
    target: float = 0.5  # compaction aims to get back under this share
    preserve_recent: int = 6
    pin_first: int = 0
    offload_tokens: int = 4000  # tool results larger than this are stored on disk; 0 turns offloading off


class ToolsConfig(BaseModel):
    enabled: bool = True
    root: Path = Path(".")  # the one project folder every tool is bound to
    notes_dir: Path = Path("notes")  # relative to root unless absolute
    shell: bool = False  # runs commands in the Sandbox with root as cwd; every command still needs TUI approval (M8)
    shell_network: bool = False  # lets sandboxed shell commands reach the network
    write: bool = True  # write_file inside root; every write needs TUI approval (M8)
    # Tools that need approval. `shell` and `write_file` are always in it, whatever the file says.
    approve: list[str] = ["shell", "write_file"]
    max_read_chars: int = 20_000

    @field_validator("approve")
    @classmethod
    def _always_gate_shell_and_write(cls, names: list[str]) -> list[str]:
        return list(dict.fromkeys([*names, "shell", "write_file"]))


class DecideConfig(BaseModel):
    """The tev1 decision layer (M4). Advisory only."""

    enabled: bool = True
    model: str = "tev1:0.8b"
    max_input_tokens: int = 1500
    keep_alive: str = "5m"
    compaction_advice_from: float = 0.8  # ask tev1 once the window is this full; the ceiling still decides alone


class TelemetryConfig(BaseModel):
    """OpenTelemetry export (M9). Off by default; when on, traces and metrics go to an OTLP/HTTP collector."""

    enabled: bool = False
    endpoint: str = "http://localhost:4318"
    service_name: str = "braid"
    metric_interval_s: float = 10.0


class EvalsConfig(BaseModel):
    """The `harness-eval` lineup (M9). Models that are not pulled are skipped and listed as such."""

    models: list[str] = ["gemma4:e4b", "gemma4:e2b", "gemma4:12b", "gpt-oss:20b"]
    num_ctx: int = 8192  # modest on purpose: the M4 prompts are short and every model must fit without spilling


class WorkflowConfig(BaseModel):
    """The Execution loop (M13) and the Plan loop (M14). ``model`` (Ollama) defaults to the Profile's model.

    ``backend = "anthropic"`` runs on the Anthropic API (ADR 0006), with ``anthropic_model``; a ``claude-`` model id
    picks it too. Credentials are never config: the SDK reads ``ANTHROPIC_API_KEY`` or an ``ant auth`` profile."""

    backend: Backend = "anthropic"  # ADR 0007; "ollama" is the frozen path
    model: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    effort: Effort | None = None  # Anthropic only; None leaves the model's default
    max_cost_usd: float = 0.10  # Spend cap: an API-backed run stops at the next node boundary once it has cost more
    num_ctx: int = 16384
    think: bool | None = False  # sent as /api/chat's top-level ``think``; false stops thinking models spending the output on it
    num_predict: int | None = 4096  # cap on generated tokens per model call
    max_attempts: int = 3  # implementation attempts, and test rewrites, per Behavior before Stuck
    show_invariants: bool = True  # F3: show the Plan's Invariants to the models (they are checked by code either way)
    check_tests_at_red: bool = True  # F3: check a new test against the Invariants after Red; off leaves only Disputes
    # F10: a stronger model (an API model id) that rules on a Dispute no Invariant upholds; None leaves the graph as in F3
    arbiter: str | None = None
    arbiter_effort: Effort | None = None  # None leaves the Arbiter model's default
    max_arbitrations: int = 1  # per Behavior; past it a Dispute no Invariant upholds takes the code route (ADR 0008)
    max_polish_rounds: int = 1  # 0 skips Polish
    test_command: str = ".venv/bin/python -m pytest -q --no-header -p no:cacheprovider"
    lint_commands: list[str] = []  # each must exit 0 after Polish, or Polish is reverted
    command_timeout_s: float = 120.0
    workspaces_dir: Path = Path("~/.braid/workspaces")
    runs_dir: Path = Path("~/.braid/runs")  # one <run_id>.jsonl per run
    plans_dir: Path = Path("~/.braid/plans")  # the Plan loop saves <plan id>.json here
    checkpoints_db: Path = Path("~/.braid/workflows.db")  # TUI runs: LangGraph checkpoints (thread_id = run id) and the run index
    max_rounds: int = 3  # Plan loop: draft/review rounds before it finalizes with "stopped at round limit"
    critique_angles: list[str] = ["feasibility", "scope", "testability", "risk", "spacing"]  # one Critique call each per round
    critique_prompts: dict[str, str] = {}  # angle -> system prompt, for a new angle or to replace a built-in one


@dataclass(frozen=True)
class RoleModel:
    """The model and effort one Role runs on."""

    model: str
    effort: Effort | None = None


def arbiter_role(wf: WorkflowConfig) -> RoleModel | None:
    """The Arbiter's model and effort (F10), or None when there is no Arbiter. The one place that reads them, so F11's
    ``[workflow.roles.arbiter]`` can take over here."""
    return RoleModel(wf.arbiter, wf.arbiter_effort) if wf.arbiter else None


class AnthropicConfig(BaseModel):
    """The Anthropic client, for the chat agent and Workflows alike (F8). The SDK retries 408, 409, 429, 5xx and
    connection errors with backoff and honors ``retry-after``; a request that still fails ends the Workflow run with
    status ``api_error`` (resumable), or the chat turn with a note."""

    max_retries: int = 2  # the SDK's default
    timeout: float = 600.0  # seconds per request (connecting has its own 5 s)
    # F9: the 5.5 models think adaptively and send the thinking back empty ("omitted") unless asked for a summary.
    # Visibility only: thinking is billed the same either way.
    thinking_display: ThinkingDisplay = "summarized"


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/.braid")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def trim_log_file(self) -> Path:
        return self.resolved_data_dir / "trims.db"

    @property
    def status_log_file(self) -> Path:
        """Debug log of every /api/ps poll and every rendered status bar state during a Workflow run."""
        return self.resolved_data_dir / "status-debug.log"

    @property
    def decisions_file(self) -> Path:
        return self.resolved_data_dir / "decisions.db"

    @property
    def sessions_file(self) -> Path:
        return self.resolved_data_dir / "sessions.db"

    @property
    def offload_dir(self) -> Path:
        return self.resolved_data_dir / "offload"

    @property
    def evals_dir(self) -> Path:
        return self.resolved_data_dir / "evals"

    @property
    def bench_ledger(self) -> Path:
        """F2: one line per benchmark run (``harness-workflow bench``), with its summary and provenance."""
        return self.resolved_data_dir / "bench.jsonl"

    @property
    def calibration_file(self) -> Path:
        return self.resolved_data_dir / "calibration.json"


class HarnessConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    profile: ProfileConfig = ProfileConfig()
    context: ContextConfig = ContextConfig()
    tools: ToolsConfig = ToolsConfig()
    decide: DecideConfig = DecideConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    evals: EvalsConfig = EvalsConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: Path | None = None) -> HarnessConfig:
    """Load config from ``path`` (default ./harness.toml); defaults if the file is absent."""
    path = path or Path("harness.toml")
    if not path.exists():
        return HarnessConfig()
    with path.open("rb") as f:
        return HarnessConfig.model_validate(tomllib.load(f))
