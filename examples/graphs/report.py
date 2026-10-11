"""Print a Markdown table of models x katas from the run summaries under <practice-dir>/*/runs/*.json.

    uv run python examples/graphs/report.py ~/projects/braid-practice [--all]

The same as ``harness-workflow report --summaries <practice-dir> [--all]``, which this calls (``workflow/bench.py``).
For repeated runs and comparisons between labels, use ``harness-workflow bench`` and ``harness-workflow report``.
"""

from __future__ import annotations

import sys

from harness.workflow.cli import main

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: report.py <practice-dir> [--all]")
    main(["report", "--summaries", args[0], *(["--all"] if "--all" in sys.argv else [])])
