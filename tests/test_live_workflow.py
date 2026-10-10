"""M13 live check against a real Ollama (HARNESS_LIVE=1): one kata through ``harness-workflow build``, hints off.

The test passes when the run reaches ``finish`` (or stops at ``stuck``) without an exception; model quality is measured by
the summary JSON and ``examples/graphs/report.py``, not gated. Override the model with HARNESS_WORKFLOW_MODEL and the kata
with HARNESS_KATA (default ~/projects/braid-practice/roman-numerals, which needs its Claude ``plan.json``).
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from harness.workflow.cli import main

pytestmark = pytest.mark.skipif(os.environ.get("HARNESS_LIVE") != "1", reason="set HARNESS_LIVE=1 to run against Ollama")


def test_one_kata_runs_to_finish_or_stuck(tmp_path: Path) -> None:
    kata = Path(os.environ.get("HARNESS_KATA", "~/projects/braid-practice/roman-numerals")).expanduser()
    if not (kata / "plan.json").exists():
        pytest.skip(f"{kata} has no plan.json (run examples/graphs/claude_plan.sh)")
    model = os.environ.get("HARNESS_WORKFLOW_MODEL", "gemma4:e4b")
    cfg = tmp_path / "h.toml"
    cfg.write_text(f'[workflow]\nworkspaces_dir = "{tmp_path}/ws"\nruns_dir = "{tmp_path}/runs"\n')
    before = {p.name for p in (kata / "runs").glob("*.json")} if (kata / "runs").exists() else set()

    def refuse(_payload: dict[str, Any]) -> str:
        return "no"  # finish asks for Approval; the live check never touches the kata

    main(["build", "--plan", str(kata / "plan.json"), "--project", str(kata), "--model", model, "--no-hints", "--config", str(cfg)], ask=refuse)
    [summary] = [p for p in (kata / "runs").glob("*.json") if p.name not in before]
    run = json.loads(summary.read_text())
    assert run["trace"][-1] in {"finish", "stuck"}
