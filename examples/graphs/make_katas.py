"""Create the five kata repos in ~/projects/braid-practice/ (M11.2). Skips any that already exist.

    uv run --no-project python examples/graphs/make_katas.py [practice-dir]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYPROJECT = """[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["."]
"""

GITIGNORE = ".venv/\n__pycache__/\n.pytest_cache/\nplan.json\nruns/\n"

KATAS: dict[str, tuple[str, str]] = {
    "roman-numerals": (
        "Convert integers to Roman numerals.",
        """`to_roman(n: int) -> str` converts an integer from 1 to 3999 to its Roman numeral, using subtractive
notation (IV, IX, XL, XC, CD, CM). Any other integer raises `ValueError`.

- `to_roman(4)` returns `"IV"`
- `to_roman(1994)` returns `"MCMXCIV"`
""",
    ),
    "bowling": (
        "Score a ten-pin bowling game.",
        """`score(rolls: list[int]) -> int` returns the total score of one complete game of ten-pin bowling, given the
pins knocked down by each roll in order. A strike scores 10 plus the next two rolls, a spare scores 10 plus the next
roll, and the tenth frame gives up to two bonus rolls. The input is always a valid, complete game.

- `score([0] * 20)` returns `0`
- `score([10] * 12)` returns `300`
""",
    ),
    "word-wrap": (
        "Wrap text to a column width.",
        """`wrap(text: str, width: int) -> str` breaks `text` into lines of at most `width` characters, joined with
`"\\n"`. Words are separated by single spaces and a line takes as many whole words as fit (greedy). A word longer than
`width` is split into `width`-sized pieces. `width` below 1 raises `ValueError`.

- `wrap("the quick brown fox", 10)` returns `"the quick\\nbrown fox"`
- `wrap("abcdefghij", 4)` returns `"abcd\\nefgh\\nij"`
""",
    ),
    "semver-compare": (
        "Compare semantic version strings.",
        """`compare(a: str, b: str) -> int` returns `-1` if version `a` is lower than `b`, `0` if they are equal and `1`
if it is higher, following semver 2.0 precedence: numeric `major.minor.patch` compared as integers, a version with a
pre-release (`1.0.0-alpha`) lower than the same version without one, pre-release identifiers compared one by one
(numeric identifiers as integers and lower than alphanumeric ones), and build metadata (`+build`) ignored. A string
that is not a valid version raises `ValueError`.

- `compare("1.2.3", "1.10.0")` returns `-1`
- `compare("1.0.0-alpha", "1.0.0")` returns `-1`
""",
    ),
    "cron-next": (
        "Find the next run time of a cron expression.",
        """`next_run(expr: str, after: datetime) -> datetime` returns the first minute strictly after `after` that matches
a five-field cron expression (minute, hour, day of month, month, day of week with Sunday as 0). Each field supports
`*`, a number, a range (`1-5`), a step (`*/15`, `1-30/5`) and comma-separated lists of those. When both day of month
and day of week are restricted, a time must match both. The result has zero seconds. A malformed expression raises
`ValueError`.

- `next_run("*/15 * * * *", datetime(2026, 1, 1, 10, 7))` returns `datetime(2026, 1, 1, 10, 15)`
- `next_run("0 9 * * *", datetime(2026, 1, 1, 9, 0))` returns `datetime(2026, 1, 2, 9, 0)`
""",
    ),
}


def run(*cmd: str, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def make(root: Path, name: str, summary: str, goal: str) -> None:
    repo = root / name
    if repo.exists():
        print(f"{name}: exists, skipped")
        return
    module = name.replace("-", "_")
    (repo / "tests").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(name=name))
    (repo / f"{module}.py").write_text(f'"""{summary}"""\n')
    (repo / "tests" / ".gitkeep").write_text("")
    (repo / ".gitignore").write_text(GITIGNORE)
    (repo / "GOAL.md").write_text(f"# {name}\n\n{goal}")
    run("git", "init", "-q", "-b", "main", cwd=repo)
    run("uv", "sync", cwd=repo)  # writes uv.lock, which the Workspace needs for `uv sync --frozen`
    run("git", "add", "-A", cwd=repo)
    run("git", "-c", "user.name=braid", "-c", "user.email=braid@localhost", "commit", "-q", "-m", "Empty kata", cwd=repo)
    print(f"{name}: created")


def main() -> None:
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "projects" / "braid-practice"
    root.mkdir(parents=True, exist_ok=True)
    for name, (summary, goal) in KATAS.items():
        make(root, name, summary, goal)


if __name__ == "__main__":
    main()
