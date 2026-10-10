"""The Workspace: a throwaway git worktree an Execution loop writes in; its diff reaches the project only via ``apply()``."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from harness.sandbox.policy import RunResult, SandboxPolicy, toolchain_paths
from harness.sandbox.seatbelt import MAX_OUTPUT_BYTES, run_process

DEFAULT_BASE = Path("~/.braid/workspaces")
SYNC_TIMEOUT_S = 600.0
RUNS_ON_IMPORT_NAMES = frozenset({"conftest.py", "sitecustomize.py", "setup.py", "setup.cfg", "pyproject.toml", "uv.lock", "Makefile"})
SYMLINK_MODE = "120000"
# The Workspace's own scratch and environment: never part of the diff.
SCRATCH = (".tmp", ".venv")
EXCLUDES = tuple(f":(exclude){name}" for name in SCRATCH)

Flag = Literal["symlink", "outside", "runs-on-import"]


@dataclass(frozen=True)
class Change:
    path: str
    status: Literal["added", "modified", "deleted"]
    flags: frozenset[Flag]


async def _git(
    cwd: Path, *args: str, env: dict[str, str] | None = None, stdin: bytes | None = None, check: bool = True,
) -> tuple[int, str]:
    """Run git in the harness (outside the Sandbox, ADR 0005); returns (exit code, stdout)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=cwd, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})},
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin)
    code = proc.returncode if proc.returncode is not None else -1
    if check and code != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {err.decode(errors='replace').strip()}")
    return code, out.decode(errors="replace")


def _flags(path: str, workspace: Path, new_mode: str) -> frozenset[Flag]:
    flags: set[Flag] = set()
    posix = PurePosixPath(path)
    if new_mode == SYMLINK_MODE:
        flags.add("symlink")
    if posix.is_absolute() or ".." in posix.parts:
        flags.add("outside")
    elif new_mode != "000000":
        real = (workspace / path).resolve()  # a file reached through a link that leaves the Workspace
        if real != workspace and workspace not in real.parents:
            flags.add("outside")
    if posix.name in RUNS_ON_IMPORT_NAMES or posix.suffix == ".pth" or posix.parts[:1] == (".github",):
        flags.add("runs-on-import")
    return frozenset(flags)


class Workspace:
    path: Path
    project: Path
    warning: str | None  # set by ``create`` when the project tree was dirty: the Workspace starts from HEAD

    def __init__(self, path: Path, project: Path, gitdir: Path, warning: str | None = None) -> None:
        self.path = path
        self.project = project
        self.warning = warning
        # Workspace git commands name the worktree's git dir outright instead of trusting the ``.git`` file inside
        # the Workspace, which model code can rewrite to point at a repository whose config runs commands.
        self._git_env = {"GIT_DIR": str(gitdir), "GIT_WORK_TREE": str(path)}

    @classmethod
    async def create(cls, project: Path, base: Path | None = None, run_id: str = "run") -> Workspace:
        project = project.resolve()
        if "/" in run_id or run_id in ("", ".", ".."):
            raise ValueError(f"bad run id {run_id!r}")
        code, top = await _git(project, "rev-parse", "--show-toplevel", check=False)
        if code != 0 or Path(top.strip()).resolve() != project:
            raise ValueError(f"{project} is not the root of a git repository")
        await _git(project, "rev-parse", "--verify", "HEAD")  # a repository with no commit has nothing to branch from
        _, status = await _git(project, "status", "--porcelain")
        warning = "the project has uncommitted changes; the Workspace starts from HEAD without them" if status.strip() else None
        root = (base or DEFAULT_BASE).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / run_id
        await _git(project, "worktree", "add", "--detach", str(path), "HEAD")
        _, gitdir = await _git(path, "rev-parse", "--absolute-git-dir")
        return cls(path.resolve(), project, Path(gitdir.strip()), warning)

    @classmethod
    async def attach(cls, project: Path, path: Path) -> Workspace:
        """The Workspace ``create`` made earlier at ``path`` (a resumed run). Its git dir is taken from the project's
        own ``.git/worktrees``, not from the ``.git`` file inside the Workspace, which model code can rewrite."""
        project, path = project.resolve(), path.resolve()
        if not path.is_dir():
            raise ValueError(f"the Workspace {path} is gone")
        _, common = await _git(project, "rev-parse", "--absolute-git-dir")
        gitdir = Path(common.strip()) / "worktrees" / path.name
        pointer = gitdir / "gitdir"
        if not pointer.is_file() or Path(pointer.read_text().strip()).parent.resolve() != path:
            raise ValueError(f"{path} is not a worktree of {project}")
        return cls(path, project, gitdir)

    async def sync_deps(self) -> RunResult:
        """``uv sync --frozen`` from the lockfile at HEAD, in the harness: before the model writes anything."""
        if not (self.path / "uv.lock").exists():
            return RunResult(0, "", False, 0.0)
        return await run_process(
            ["uv", "sync", "--frozen"], cwd=self.path, env=dict(os.environ), timeout_s=SYNC_TIMEOUT_S,
            max_output_bytes=MAX_OUTPUT_BYTES,
        )

    def policy(self) -> SandboxPolicy:
        """Writes in the Workspace and its ``.tmp`` only. The project's ``.git`` stays unwritable (and, under the
        home folder, unreadable), so model code cannot plant a hook that later runs unsandboxed."""
        venv = self.path / ".venv"
        return SandboxPolicy(
            writable=(self.path, self.path / ".tmp"), readable=toolchain_paths(venv) if venv.exists() else (),
            network=False,
        )

    async def _intent_to_add(self) -> None:
        """Make new files visible to ``git diff HEAD``. Done by listing them: ``add`` with an exclude pathspec
        fails when the excluded folder is also gitignored."""
        _, listing = await _git(self.path, "ls-files", "--others", "--exclude-standard", "-z", env=self._git_env)
        new = [f for f in listing.split("\0") if f and f.split("/")[0] not in SCRATCH]
        if new:
            await _git(
                self.path, "add", "--intent-to-add", "--pathspec-from-file=-", "--pathspec-file-nul",
                env={**self._git_env, "GIT_LITERAL_PATHSPECS": "1"}, stdin="\0".join(new).encode() + b"\0",
            )

    async def list_files(self) -> list[str]:
        """Tracked and new (not ignored) files, sorted: what ``git ls-files`` shows, minus the Workspace's own scratch.
        Tracked files deleted from disk are left out."""
        _, listing = await _git(self.path, "ls-files", "--cached", "--others", "--exclude-standard", "-z", env=self._git_env)
        names = {f for f in listing.split("\0") if f and f.split("/")[0] not in SCRATCH}
        return sorted(f for f in names if (self.path / f).is_file() and not (self.path / f).is_symlink())

    async def read_head(self, path: str) -> bytes | None:
        """A file's contents at HEAD (the Workspace's starting point), or None when HEAD has no such file."""
        proc = await asyncio.create_subprocess_exec(
            "git", "show", f"HEAD:{path}", cwd=self.path, env={**os.environ, **self._git_env, "GIT_LITERAL_PATHSPECS": "1"},
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return out if proc.returncode == 0 else None

    async def changes(self) -> list[Change]:
        await self._intent_to_add()
        _, raw = await _git(
            self.path, "diff", "HEAD", "--raw", "-z", "--no-renames", "--no-ext-diff", "--", ".", *EXCLUDES,
            env=self._git_env,
        )
        fields = raw.split("\0")
        found: list[Change] = []
        for meta, path in zip(fields[0::2], fields[1::2], strict=False):
            _, new_mode, _, _, letter = meta.lstrip(":").split(" ")
            status: Literal["added", "modified", "deleted"] = {"A": "added", "D": "deleted"}.get(letter[0], "modified")  # type: ignore[assignment]
            found.append(Change(path, status, _flags(path, self.path, new_mode)))
        return found

    async def diff(self) -> str:
        await self._intent_to_add()
        _, text = await _git(
            self.path, "diff", "HEAD", "--binary", "--no-renames", "--no-ext-diff", "--no-textconv", "--", ".", *EXCLUDES,
            env=self._git_env,
        )
        return text

    async def apply(self) -> None:
        """Apply the Workspace diff to the project's working tree; never commits. Refuses flagged links and paths, and
        any project file that no longer matches HEAD (the user's edits are not overwritten)."""
        changes = await self.changes()
        for change in changes:
            if change.flags & {"symlink", "outside"}:
                raise ValueError(f"refusing to apply {change.path}: {', '.join(sorted(change.flags & {'symlink', 'outside'}))}")
        for change in changes:
            _, dirty = await _git(self.project, "status", "--porcelain", "--", change.path)
            if dirty.strip() or (change.status == "added" and (self.project / change.path).exists()):
                raise ValueError(f"refusing to apply: {change.path} in the project differs from HEAD")
        patch = await self.diff()
        if patch:
            await _git(self.project, "apply", "--binary", "-", stdin=patch.encode())

    async def remove(self) -> None:
        await _git(self.project, "worktree", "remove", "--force", str(self.path), check=False)
        shutil.rmtree(self.path, ignore_errors=True)
        await _git(self.project, "worktree", "prune", check=False)
