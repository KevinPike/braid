import sys
from pathlib import Path

import pytest

from harness.sandbox import SeatbeltSandbox, Workspace
from tests.conftest import git


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "workspaces"


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.parts}


async def test_new_and_edited_files_reach_the_project_only_through_apply(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r1")
    before = tree(fixture_repo)
    (ws.path / "src" / "new.py").write_text("X = 1\n")
    (ws.path / "src" / "mod.py").write_text("def add(a: int, b: int) -> int:\n    return b + a\n")
    changes = {c.path: c for c in await ws.changes()}
    assert {p: c.status for p, c in changes.items()} == {"src/new.py": "added", "src/mod.py": "modified"}
    diff = await ws.diff()
    assert "+X = 1" in diff and "return b + a" in diff
    assert tree(fixture_repo) == before and not (fixture_repo / "src" / "new.py").exists()
    await ws.apply()
    assert tree(fixture_repo) == tree(ws.path)
    assert git(fixture_repo, "log", "--oneline").count("\n") == 1  # apply never commits
    await ws.remove()


async def test_a_symlink_that_leaves_the_workspace_makes_apply_raise(fixture_repo: Path, base: Path, tmp_path: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r2")
    (ws.path / "src" / "mod.py").write_text("changed = True\n")
    (ws.path / "link").symlink_to(tmp_path)
    flags = {c.path: c.flags for c in await ws.changes()}
    assert {"symlink", "outside"} <= flags["link"]
    before = tree(fixture_repo)
    with pytest.raises(ValueError, match="link"):
        await ws.apply()
    assert tree(fixture_repo) == before and not (fixture_repo / "link").exists()
    await ws.remove()


async def test_an_edited_conftest_carries_the_runs_on_import_flag(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r3")
    (ws.path / "tests" / "conftest.py").write_text("import os\n")
    (ws.path / "src" / "plain.py").write_text("")
    (ws.path / "evil.pth").write_text("import os\n")
    flags = {c.path: c.flags for c in await ws.changes()}
    assert "runs-on-import" in flags["tests/conftest.py"] and "runs-on-import" in flags["evil.pth"]
    assert "runs-on-import" not in flags["src/plain.py"]
    await ws.remove()


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
async def test_model_code_in_the_workspace_cannot_write_a_project_git_hook(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r4")
    hook = fixture_repo / ".git" / "hooks" / "pre-commit"
    cmd = f"echo '#!/bin/sh' > {hook}; echo ok > ok.txt"
    result = await SeatbeltSandbox().run(cmd, cwd=ws.path, policy=ws.policy())
    assert not hook.exists() and "Operation not permitted" in result.output
    assert (ws.path / "ok.txt").exists()  # writes inside the Workspace still work
    await ws.remove()


async def test_remove_leaves_no_worktree(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r5")
    assert "r5" in git(fixture_repo, "worktree", "list")
    await ws.remove()
    assert "r5" not in git(fixture_repo, "worktree", "list") and not ws.path.exists()


async def test_create_refuses_a_non_git_folder(tmp_path: Path, base: Path) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError, match="not the root of a git repository"):
        await Workspace.create(tmp_path / "plain", base, "x")


async def test_dirty_project_warns_and_apply_refuses_files_that_differ_from_head(fixture_repo: Path, base: Path) -> None:
    (fixture_repo / "src" / "mod.py").write_text("mine = 1\n")
    ws = await Workspace.create(fixture_repo, base, "r6")
    assert ws.warning and "uncommitted" in ws.warning
    assert "mine" not in (ws.path / "src" / "mod.py").read_text()  # starts from HEAD
    (ws.path / "src" / "mod.py").write_text("theirs = 1\n")
    with pytest.raises(ValueError, match="differs from HEAD"):
        await ws.apply()
    assert (fixture_repo / "src" / "mod.py").read_text() == "mine = 1\n"
    await ws.remove()


async def test_scratch_dirs_are_not_part_of_the_diff_and_sync_is_a_no_op_without_a_lockfile(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r7")
    (ws.path / ".tmp").mkdir()
    (ws.path / ".tmp" / "junk").write_text("x")
    assert await ws.changes() == [] and await ws.diff() == ""
    assert (await ws.sync_deps()).exit_code == 0
    await ws.remove()


async def test_odd_file_names_are_added_literally(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r8")
    (ws.path / "a*b.txt").write_text("x")
    (ws.path / "other.txt").write_text("y")
    assert {c.path for c in await ws.changes()} == {"a*b.txt", "other.txt"}
    await ws.remove()


async def test_list_files_and_read_head_see_new_files_and_the_starting_contents(fixture_repo: Path, base: Path) -> None:
    ws = await Workspace.create(fixture_repo, base, "r6")
    (ws.path / "src" / "mod.py").write_text("changed = True\n")
    (ws.path / "src" / "new.py").write_text("X = 1\n")
    (ws.path / ".venv").mkdir()
    (ws.path / ".venv" / "junk.py").write_text("")
    assert await ws.list_files() == [".gitignore", "pyproject.toml", "src/mod.py", "src/new.py", "tests/test_mod.py"]
    assert (await ws.read_head("src/mod.py") or b"").startswith(b"def add")
    assert await ws.read_head("src/new.py") is None
    await ws.remove()
