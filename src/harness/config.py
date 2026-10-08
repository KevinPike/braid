"""Loads harness.toml: the model profile and paths."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


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


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/.harness")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def trim_log_file(self) -> Path:
        return self.resolved_data_dir / "trims.db"

    @property
    def offload_dir(self) -> Path:
        return self.resolved_data_dir / "offload"

    @property
    def calibration_file(self) -> Path:
        return self.resolved_data_dir / "calibration.json"


class HarnessConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    profile: ProfileConfig = ProfileConfig()
    context: ContextConfig = ContextConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: Path | None = None) -> HarnessConfig:
    """Load config from ``path`` (default ./harness.toml); defaults if the file is absent."""
    path = path or Path("harness.toml")
    if not path.exists():
        return HarnessConfig()
    with path.open("rb") as f:
        return HarnessConfig.model_validate(tomllib.load(f))
