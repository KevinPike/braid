"""Fixtures shared by the Sandbox, Workspace and (from M13) Execution loop tests: a fixture git repo and a venv with pytest."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo: Path, *args: str) -> str:
    import os

    done = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env={**os.environ, **GIT_ENV},
    )
    return done.stdout


def make_fixture_repo(root: Path) -> Path:
    """A small committed git project: a ``pyproject.toml``, one module and a ``tests/`` folder."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "0.0.0"\n')
    (root / "src").mkdir()
    (root / "src" / "mod.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_mod.py").write_text("from src.mod import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n")
    (root / ".gitignore").write_text(".venv/\n.tmp/\n__pycache__/\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture")
    return root


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    return make_fixture_repo(tmp_path / "project")


@pytest.fixture(scope="session")
def fixture_venv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A venv with pytest installed, built once per session; the test is skipped when uv cannot build it."""
    venv = tmp_path_factory.mktemp("venv") / ".venv"
    steps = [["uv", "venv", "--python", "3.12", str(venv)]]
    pip = ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "pytest"]
    try:
        for step in steps:
            subprocess.run([*step, "--offline"], check=True, capture_output=True)
        try:
            subprocess.run([*pip, "--offline"], check=True, capture_output=True)  # from the uv cache
        except subprocess.CalledProcessError:
            subprocess.run(pip, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"cannot build a fixture venv with pytest (needs uv, and its cache or network): {exc}")
    return venv
