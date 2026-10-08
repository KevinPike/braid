from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.tools.builtin import ProjectSandbox, build_shell, build_tools, resolve_inside


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "README.md").write_text("# Hello\nsecond line\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "garden.md").write_text("Plant tomatoes\nWater the Dentist reminder? no\n")
    (tmp_path / "notes" / "work.txt").write_text("Q3 budget is final\n")
    return tmp_path


def tools_for(project: Path, **kw: object):  # type: ignore[no-untyped-def]
    ts = build_tools(project, Path("notes"), now=lambda: datetime(2026, 10, 8, 12, 30, tzinfo=timezone.utc), **kw)  # type: ignore[arg-type]
    return {t.tool_name: t for t in ts}


def test_four_tools_with_schemas(project: Path) -> None:
    ts = tools_for(project)
    assert set(ts) == {"read_file", "list_directory", "search_notes", "current_time"}
    assert all(t.tool_spec["description"] for t in ts.values())


def test_read_file_and_truncation(project: Path) -> None:
    assert tools_for(project)["read_file"]("README.md").startswith("# Hello")
    out = tools_for(project, max_read_chars=5)["read_file"]("README.md")
    assert out.startswith("# Hel") and "truncated" in out


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "src/../../x"])
def test_paths_outside_the_project_are_refused(project: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="outside the project folder"):
        tools_for(project)["read_file"](bad)


def test_symlink_escape_is_refused(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "s.txt").write_text("secret")
    (project / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_inside(project, "link/s.txt")


def test_list_directory_puts_folders_first(project: Path) -> None:
    assert tools_for(project)["list_directory"](".").splitlines()[:2] == ["notes/", "src/"]
    assert tools_for(project)["list_directory"]("src") == "a.py"


def test_search_notes_is_case_insensitive_and_reports_misses(project: Path) -> None:
    ts = tools_for(project)
    assert "garden.md:2:" in ts["search_notes"]("dentist")
    assert "work.txt:1:" in ts["search_notes"]("q3 BUDGET")
    assert "no notes mention" in ts["search_notes"]("kubernetes")


def test_current_time_uses_the_requested_zone(project: Path) -> None:
    ts = tools_for(project)
    assert "2026-10-08 12:30:00 UTC" in ts["current_time"]()
    assert "14:30:00" in ts["current_time"]("Europe/Lisbon") or "13:30:00" in ts["current_time"]("Europe/Lisbon")
    with pytest.raises(ValueError, match="unknown timezone"):
        ts["current_time"]("Mars/Base")


async def test_shell_runs_in_the_project_folder(project: Path) -> None:
    result = await ProjectSandbox(project).execute("pwd")
    assert result.stdout.strip() == str(project.resolve())
    assert build_shell(project).tool_name == "shell"
