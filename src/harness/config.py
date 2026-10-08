"""Loads harness.toml: the model profile and paths."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"


class ProfileConfig(BaseModel):
    model: str = "gemma4:e4b"
    num_ctx: int = 32768
    keep_alive: str = "30m"
    system_prompt: str = "You are a concise, helpful assistant running locally."


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/.harness")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser()


class HarnessConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    profile: ProfileConfig = ProfileConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: Path | None = None) -> HarnessConfig:
    """Load config from ``path`` (default ./harness.toml); defaults if the file is absent."""
    path = path or Path("harness.toml")
    if not path.exists():
        return HarnessConfig()
    with path.open("rb") as f:
        return HarnessConfig.model_validate(tomllib.load(f))
