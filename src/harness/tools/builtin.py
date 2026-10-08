"""Read-only tools plus the optional file write and shell, all bound to one project folder."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from strands import tool
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.sandbox.types import ExecutionResult, StreamChunk
from strands.tools.decorator import DecoratedFunctionTool
from strands.vended_tools.shell import make_shell

NOTE_SUFFIXES = {".md", ".txt", ".org"}
MAX_SEARCH_HITS = 20
MAX_LIST_ENTRIES = 200


def resolve_inside(root: Path, path: str) -> Path:
    """Resolve ``path`` against ``root``; anything that lands outside it (``..``, absolute, symlink) is refused."""
    base = root.resolve()
    target = (base / path).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"{path!r} is outside the project folder")
    return target


def build_tools(
    root: Path,
    notes_dir: Path,
    *,
    max_read_chars: int = 20_000,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> list[DecoratedFunctionTool[Any, Any]]:
    root = root.resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the project folder and return its contents.

        Args:
            path: File path relative to the project folder.
        """
        target = resolve_inside(root, path)
        if not target.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_read_chars:
            return text[:max_read_chars] + f"\n[truncated: {len(text) - max_read_chars} more characters]"
        return text

    @tool
    def list_directory(path: str = ".") -> str:
        """List the files and folders in a directory of the project folder.

        Args:
            path: Directory path relative to the project folder.
        """
        target = resolve_inside(root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"[{len(entries) - MAX_LIST_ENTRIES} more entries not shown]")
        return "\n".join(lines) or "(empty)"

    @tool
    def search_notes(query: str) -> str:
        """Search the user's notes for a word or phrase (case-insensitive) and return matching lines.

        Args:
            query: Text to look for.
        """
        if notes_dir.is_absolute():
            folder = notes_dir.resolve()  # the one place the user may point outside the project
        else:
            try:
                folder = resolve_inside(root, str(notes_dir))
            except ValueError:
                return f"notes folder {notes_dir} is outside the project folder"
        if not folder.is_dir():
            return f"no notes folder at {folder}"
        needle = query.lower()
        hits: list[str] = []
        for file in sorted(folder.rglob("*")):
            if file.suffix.lower() not in NOTE_SUFFIXES:
                continue
            relative = file.relative_to(folder)
            try:
                real = resolve_inside(folder, str(relative))  # a symlink leading out of the notes folder is skipped
            except ValueError:
                continue
            if not real.is_file():
                continue
            for number, line in enumerate(real.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    hits.append(f"{relative}:{number}: {line.strip()}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits) + f"\n[stopped at {MAX_SEARCH_HITS} matches]"
        return "\n".join(hits) or f"no notes mention {query!r}"

    @tool
    def current_time(timezone_name: str = "UTC") -> str:
        """Return the current date and time.

        Args:
            timezone_name: IANA timezone such as UTC or Europe/Lisbon.
        """
        try:
            zone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {timezone_name!r}") from exc
        return now().astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")

    return [read_file, list_directory, search_notes, current_time]


def build_write_file(root: Path) -> DecoratedFunctionTool[Any, Any]:
    """The file write tool. It changes the user's files, so the app only registers it behind the approval gate."""
    root = root.resolve()

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file in the project folder, replacing it if it exists. Needs the user's approval.

        Args:
            path: File path relative to the project folder; missing parent folders are created.
            content: The full text to write.
        """
        target = resolve_inside(root, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)  # resolve_inside already put target, so its parents, inside root
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {target.relative_to(root)}"

    return write_file


class ProjectSandbox(NotASandboxLocalEnvironment):
    """Runs every command with the project folder as its working directory.

    This is a working directory, not isolation: a command can still ``cd ..`` or use absolute
    paths. The boundary that matters is the TUI approval prompt (M8, ADR 0003): the app registers the
    shell only behind the approval gate, and only when ``[tools] shell = true``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    async def execute_streaming(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None,
        env: dict[str, str] | None = None, **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        async for chunk in super().execute_streaming(command, timeout=timeout, cwd=str(self.root), env=env, **kwargs):
            yield chunk


def build_shell(root: Path) -> DecoratedFunctionTool[Any, Any]:
    return make_shell(sandbox=ProjectSandbox(root))

