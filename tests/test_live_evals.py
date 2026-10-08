"""M9 gate against a real Ollama (HARNESS_LIVE=1): the eval command on one pulled model, three prompts."""

import os
from pathlib import Path

import pytest

from harness.evals.cli import main

pytestmark = pytest.mark.skipif(os.environ.get("HARNESS_LIVE") != "1", reason="set HARNESS_LIVE=1 to run against Ollama")


def test_eval_command_on_one_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("strands_evals", reason="optional `evals` dependency group not installed")
    model = os.environ.get("HARNESS_DRIVER", "gemma4:e4b")
    cfg = tmp_path / "h.toml"
    cfg.write_text(f'[paths]\ndata_dir = "{tmp_path}"\n')
    main(["--config", str(cfg), "--models", model, "--limit", "3"])
    out = capsys.readouterr().out
    print(out)
    assert f"| {model} |" in out
    assert len(list((tmp_path / "evals").glob("*.md"))) == 1 and len(list((tmp_path / "evals").glob("*.json"))) == 1
