"""Loads harness.toml: the model profile and paths."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"


class ProfileConfig(BaseModel):
    model: str = "gemma4:e4b"
    num_ctx: int = 32768
    keep_alive: str = "30m"
    system_prompt: str = "You are a concise, helpful assistant running locally."
    # Tried in order when `model` does not fit the GPU budget (preflight step-down).
    fallbacks: list[str] = []


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


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/.braid")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def trim_log_file(self) -> Path:
        return self.resolved_data_dir / "trims.db"

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
    paths: PathsConfig = PathsConfig()


def load_config(path: Path | None = None) -> HarnessConfig:
    """Load config from ``path`` (default ./harness.toml); defaults if the file is absent."""
    path = path or Path("harness.toml")
    if not path.exists():
        return HarnessConfig()
    with path.open("rb") as f:
        return HarnessConfig.model_validate(tomllib.load(f))
