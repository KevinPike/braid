import socket
import sys
from datetime import datetime, UTC
from pathlib import Path

import pytest

from harness.tools.builtin import ProjectSandbox, build_shell, build_tools, build_write_file, resolve_inside


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
    ts = build_tools(project, Path("notes"), now=lambda: datetime(2026, 10, 8, 12, 30, tzinfo=UTC), **kw)  # type: ignore[arg-type]
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


darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")


@darwin_only
async def test_shell_runs_in_the_project_folder(project: Path) -> None:
    result = await ProjectSandbox(project).execute("pwd")
    assert result.stdout.strip() == str(project.resolve())
    assert build_shell(project).tool_name == "shell"


@darwin_only
async def test_an_approved_shell_command_cannot_write_outside_the_root(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    result = await ProjectSandbox(project).execute(f"echo x > {outside}/escape.txt; echo y > {project}/inside.txt")
    assert "Operation not permitted" in result.stderr + result.stdout
    assert not (outside / "escape.txt").exists()
    assert (project / "inside.txt").read_text() == "y\n"


@darwin_only
async def test_shell_env_is_scrubbed_and_network_follows_the_option(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAID_TEST_SECRET", "leaky")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    cmd = f"echo \"[$BRAID_TEST_SECRET]\"; curl -sS -m 3 http://127.0.0.1:{port}/ >/dev/null 2>&1; echo curl=$?"
    try:
        off = await ProjectSandbox(project).execute(cmd)
        assert off.stdout.splitlines() == ["[]", "curl=7"]  # 7: could not connect
        on = await ProjectSandbox(project, network=True).execute(cmd)
        assert on.stdout.splitlines()[1] != "curl=7"
    finally:
        server.close()


@darwin_only
async def test_shell_timeout_raises_the_strands_error(project: Path) -> None:
    from strands.sandbox.errors import SandboxTimeoutError

    with pytest.raises(SandboxTimeoutError):
        await ProjectSandbox(project).execute("sleep 30", timeout=0.5)


def test_shell_network_defaults_off() -> None:
    from harness.config import ToolsConfig

    assert ToolsConfig().shell_network is False


def test_search_notes_skips_symlinks_that_leave_the_notes_folder(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.md").write_text("hunter2 is the password\n")
    (project / "notes" / "leak.md").symlink_to(outside / "secret.md")
    (project / "notes" / "inside.md").symlink_to(project / "notes" / "work.txt")  # a link that stays inside is fine
    ts = tools_for(project)
    assert "no notes mention" in ts["search_notes"]("hunter2")
    assert "inside.md:1:" in ts["search_notes"]("budget")


def test_search_notes_refuses_a_relative_notes_folder_that_escapes_the_project(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "a.md").write_text("secret\n")
    (project / "linked").symlink_to(outside)
    ts = {t.tool_name: t for t in build_tools(project, Path("linked"))}
    assert "outside the project folder" in ts["search_notes"]("secret")
    ts = {t.tool_name: t for t in build_tools(project, Path("../elsewhere"))}
    assert "outside the project folder" in ts["search_notes"]("secret")


def test_an_absolute_notes_folder_is_allowed_to_live_elsewhere(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "a.md").write_text("findme\n")
    ts = {t.tool_name: t for t in build_tools(project, elsewhere)}
    assert "a.md:1:" in ts["search_notes"]("findme")


def test_write_file_writes_inside_the_project_and_creates_parents(project: Path) -> None:
    write = build_write_file(project)
    assert write.tool_name == "write_file" and write.tool_spec["description"]
    assert "wrote 5 characters to docs/new/a.txt" in write("docs/new/a.txt", "hello")
    assert (project / "docs" / "new" / "a.txt").read_text() == "hello"
    write("docs/new/a.txt", "again")  # replaces
    assert (project / "docs" / "new" / "a.txt").read_text() == "again"


@pytest.mark.parametrize("bad", ["../escape.txt", "/tmp/escape.txt", "src/../../escape.txt", "../newdir/x.txt"])
def test_write_file_refuses_paths_outside_the_project(project: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="outside the project folder"):
        build_write_file(project)(bad, "x")
    assert not (project.parent / "escape.txt").exists() and not (project.parent / "newdir").exists()


def test_write_file_refuses_symlinks_that_leave_the_project_and_makes_no_folders_there(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (project / "link").symlink_to(outside)
    (project / "filelink").symlink_to(outside / "target.txt")
    write = build_write_file(project)
    for bad in ("link/x.txt", "link/newdir/x.txt", "filelink"):
        with pytest.raises(ValueError, match="outside the project folder"):
            write(bad, "x")
    assert list(outside.iterdir()) == []


def test_write_file_refuses_a_directory(project: Path) -> None:
    with pytest.raises(IsADirectoryError):
        build_write_file(project)("src", "x")
    with pytest.raises(IsADirectoryError):
        build_write_file(project)(".", "x")
