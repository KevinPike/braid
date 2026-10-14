"""``run_skill_script`` (F15): a skill's ``scripts/`` run only through the ``Sandbox``, behind an Approval every time.

The tool is in ``ALWAYS_GATED`` (approval.py), so it never runs on advice or on a skill's own say-so, whatever the skill's
``allowed-tools`` claims (ADR 0011). The script gets the project writable, the skill's directory readable, no network
unless the model asks for it (a Sandbox escape the Approval shows), no stdin, and an environment built from scratch: the
harness's own environment, API keys included, never reaches it.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from strands import tool
from strands.tools.decorator import DecoratedFunctionTool

from harness.sandbox import Sandbox, SandboxPolicy, toolchain_paths
from harness.skills.catalog import SkillCatalog

SCRIPT_TOOL = "run_skill_script"
INTERPRETERS = {".py": "python3", ".sh": "sh"}  # any other script must be executable and carry a shebang


def script_command(script: Path, args: list[str]) -> str:
    interpreter = INTERPRETERS.get(script.suffix)
    parts = [interpreter, str(script), *args] if interpreter else [str(script), *args]
    return " ".join(shlex.quote(p) for p in parts)


def build_run_skill_script(
    catalog: SkillCatalog, sandbox: Sandbox, root: Path, *, timeout_s: float = 60.0,
) -> DecoratedFunctionTool[Any, Any]:
    root = root.resolve()

    @tool
    async def run_skill_script(name: str, script: str, args: list[str] | None = None, network: bool = False) -> str:
        """Run a script from a skill's scripts/ folder in the Sandbox. Needs the user's approval each time.

        Args:
            name: The skill's name.
            script: The script's path inside the skill, for example scripts/report.py.
            args: Command-line arguments, one string each.
            network: True only if the script cannot work without the network; the user is asked to allow that.
        """
        try:
            path = catalog.resource_path(name, script)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        skill_dir = catalog.entry(name).dir
        if path.relative_to(skill_dir).parts[0] != "scripts":
            raise ValueError(f"{script!r} is not under {name}/scripts/")
        venv = root / ".venv"
        bin_dirs = [str(venv / "bin")] if venv.exists() else []
        env_path = os.pathsep.join([*bin_dirs, os.environ.get("PATH", "")])
        policy = SandboxPolicy(
            writable=(root, root / ".tmp"), readable=(skill_dir, *(toolchain_paths(venv) if venv.exists() else ())), network=network,
        )
        # PATH is the only thing taken from the harness's environment (it is not a secret). The Sandbox adds HOME, LANG and
        # TERM; nothing else, and so no *_API_KEY, is passed on.
        result = await sandbox.run(
            script_command(path, list(args or [])), cwd=root, policy=policy, timeout_s=timeout_s,
            env={"PATH": env_path, "BRAID_SKILL_DIR": str(skill_dir)},
        )
        status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
        return f"{status}\n{result.output}".rstrip()

    return run_skill_script
