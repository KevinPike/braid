"""Research without tool calling (14.1): the harness runs the model's lookups in-process and checks every citation."""

from __future__ import annotations

from pathlib import Path

from harness.workflow.research import (
    Finding, Findings, ResearchQueries, ResearchQuery, check_citation, research, run_query, tracked_files,
)
from tests.workflow.fakes import SchemaModel


def q(kind: str, target: str) -> ResearchQuery:
    return ResearchQuery.model_validate({"kind": kind, "target": target, "why": "because"})


def test_read_returns_numbered_lines_capped_at_300(fixture_repo: Path) -> None:
    (fixture_repo / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 401)))
    r = run_query(fixture_repo, q("read", "big.txt"), [])
    lines = r.text.splitlines()
    assert lines[0] == "1: line 1" and lines[299] == "300: line 300" and "100 more lines" in lines[300] and len(lines) == 301


def test_grep_searches_the_tracked_files_and_stops_at_50_hits(fixture_repo: Path) -> None:
    hit = run_query(fixture_repo, q("grep", r"def add\("), tracked_files(fixture_repo))
    assert hit.text == "src/mod.py:1: def add(a: int, b: int) -> int:"
    (fixture_repo / "many.txt").write_text("needle\n" * 80)
    many = run_query(fixture_repo, q("grep", "needle"), tracked_files(fixture_repo))
    assert len(many.text.splitlines()) == 51 and "stopped at 50 hits" in many.text
    assert "not a valid regular expression" in run_query(fixture_repo, q("grep", "("), []).refused


def test_list_names_a_directory_and_marks_subdirectories(fixture_repo: Path) -> None:
    assert run_query(fixture_repo, q("list", "."), []).text.splitlines() == [".gitignore", "pyproject.toml", "src/", "tests/"]
    assert run_query(fixture_repo, q("list", "src"), []).text == "mod.py"


def test_a_target_outside_the_project_is_refused_and_not_read(fixture_repo: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("top secret")
    for target in ("../secret.txt", str(tmp_path / "secret.txt")):
        r = run_query(fixture_repo, q("read", target), [])
        assert r.refused and "outside the project" in r.refused and "top secret" not in r.text
    assert "outside the project" in run_query(fixture_repo, q("list", ".."), []).refused


def test_a_tracked_symlink_to_the_outside_is_not_searched(fixture_repo: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("needle in the outside")
    (fixture_repo / "link.txt").symlink_to(tmp_path / "secret.txt")
    assert run_query(fixture_repo, q("grep", "needle"), ["link.txt"]).text == "(no matches)"
    assert run_query(fixture_repo, q("read", "link.txt"), []).refused


def test_citations_are_checked_for_file_and_line_range(fixture_repo: Path) -> None:
    assert check_citation(fixture_repo, "src/mod.py:2") == ""
    assert check_citation(fixture_repo, "src/mod.py:1-2 - add returns the sum") == ""
    assert "does not exist" in check_citation(fixture_repo, "src/nope.py:1")
    assert "has 2 lines" in check_citation(fixture_repo, "src/mod.py:99")
    assert "path:line" in check_citation(fixture_repo, "somewhere in the code")
    assert "outside" in check_citation(fixture_repo, "../x.py:1")


async def test_research_makes_two_calls_with_the_lookups_between_and_marks_bad_citations(fixture_repo: Path) -> None:
    model = SchemaModel({
        "ResearchQueries": [ResearchQueries(queries=[q("read", "src/mod.py"), q("read", "../etc/passwd")])],
        "Findings": [Findings(findings=[
            Finding(claim="add sums", evidence="src/mod.py:2"), Finding(claim="mul exists", evidence="src/mod.py:40"),
        ])],
    })
    out = await research(model, fixture_repo, "add mul", feedback="look at tests")
    assert [c.schema for c in model.calls] == ["ResearchQueries", "Findings"]
    assert "look at tests" in model.calls[0].prompt and "src/mod.py" in model.calls[0].prompt
    assert "1: def add" in model.calls[1].prompt and "REFUSED" in model.calls[1].prompt  # the lookups reach call two
    assert [f.verified for f in out.findings] == [True, False]
    assert len(out.questions) == 1 and "mul exists" in out.questions[0] and "src/mod.py:40" in out.questions[0]
    assert out.model_calls == 2
