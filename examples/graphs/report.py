"""Print a Markdown table of models x katas from the run summaries under <practice-dir>/*/runs/*.json.

    uv run --no-project python examples/graphs/report.py ~/projects/braid-practice [--all]

Per cell: Behaviors Green / planned, attempts, Red rejections, hints, minutes per Behavior.
By default only the latest --no-hints run of each model x kata counts; --all includes hinted runs too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(practice: Path, include_hinted: bool) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(practice.glob("*/runs/*.json")):  # names start with a timestamp, so later wins
        run = json.loads(path.read_text())
        if run.get("no_hints") or include_hinted:
            latest[(run["model"], run["kata"])] = run
    return latest


def cell(run: dict[str, Any] | None) -> str:
    if run is None:
        return "-"
    t = run["totals"]
    minutes = t["seconds"] / 60 / max(run["planned"], 1)
    return f"{t['green']}/{run['planned']} · {t['attempts']}a · {t['red_rejections']}r · {t['hints']}h · {t.get('satisfied', 0)}s · {minutes:.1f}m"


def table(runs: dict[tuple[str, str], dict[str, Any]]) -> str:
    models = sorted({m for m, _ in runs})
    katas = sorted({k for _, k in runs})
    lines = ["| model | " + " | ".join(katas) + " | total Green |", "|---|" + "---|" * (len(katas) + 1)]
    for model in models:
        got = [runs.get((model, k)) for k in katas]
        green = sum(r["totals"]["green"] for r in got if r)
        planned = sum(r["planned"] for r in got if r)
        lines.append(f"| {model} | " + " | ".join(cell(r) for r in got) + f" | {green}/{planned} |")
    lines.append("")
    lines.append("Cell: Behaviors Green/planned · impl attempts (a) · Red rejections (r) · hints (h) · Behaviors already satisfied by earlier code (s, counted as Green) · minutes per Behavior (m).")
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: report.py <practice-dir> [--all]")
    runs = load(Path(args[0]).expanduser(), "--all" in sys.argv)
    print(table(runs) if runs else "no runs found")


if __name__ == "__main__":
    main()
