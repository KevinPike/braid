import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from harness.sandbox import LocalSandbox, SandboxPolicy, SeatbeltSandbox, toolchain_paths
from harness.sandbox.seatbelt import render_profile

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")

PY = sys.executable


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    work = (tmp_path / "work").resolve()
    secret_dir = (tmp_path / "secret").resolve()
    work.mkdir()
    secret_dir.mkdir()
    (secret_dir / "s.txt").write_text("hunter2")
    (work / "ok.txt").write_text("fine")
    return work, secret_dir


def accept_once(server: socket.socket) -> None:
    try:
        server.accept()
    except OSError:
        pass  # closed by the test


def policy(work: Path, secret_dir: Path, **kw: object) -> SandboxPolicy:
    # sys.executable's prefix is readable so the test interpreter starts whatever it is
    return SandboxPolicy(writable=(work, work / ".tmp"), readable=(Path(sys.base_prefix),), denied_read=(secret_dir,), **kw)  # type: ignore[arg-type]


async def test_denied_read_is_blocked_and_writable_read_works(dirs: tuple[Path, Path]) -> None:
    work, secret_dir = dirs
    sb = SeatbeltSandbox()
    bad = await sb.run(f"{PY} -c \"print(open('{secret_dir}/s.txt').read())\"", cwd=work, policy=policy(work, secret_dir))
    assert bad.exit_code != 0 and "hunter2" not in bad.output and "Operation not permitted" in bad.output
    good = await sb.run(f"{PY} -c \"print(open('{work}/ok.txt').read())\"", cwd=work, policy=policy(work, secret_dir))
    assert good.exit_code == 0 and good.output.strip() == "fine"


async def test_writes_only_inside_writable(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    work, secret_dir = dirs
    outside = tmp_path / "outside.txt"
    sb = SeatbeltSandbox()
    bad = await sb.run(f"{PY} -c \"open('{outside}', 'w').write('x')\"", cwd=work, policy=policy(work, secret_dir))
    assert bad.exit_code != 0 and not outside.exists()
    good = await sb.run(f"{PY} -c \"open('{work}/new.txt', 'w').write('x')\"", cwd=work, policy=policy(work, secret_dir))
    assert good.exit_code == 0 and (work / "new.txt").read_text() == "x"


async def test_network_only_when_the_policy_allows_it(dirs: tuple[Path, Path]) -> None:
    work, secret_dir = dirs
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    threading.Thread(target=accept_once, args=(server,), daemon=True).start()
    cmd = f"{PY} -c \"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3); print('connected')\""
    try:
        sb = SeatbeltSandbox()
        off = await sb.run(cmd, cwd=work, policy=policy(work, secret_dir, network=False))
        assert off.exit_code != 0 and "connected" not in off.output
        on = await sb.run(cmd, cwd=work, policy=policy(work, secret_dir, network=True))
        assert on.exit_code == 0 and "connected" in on.output
    finally:
        server.close()


async def test_environment_is_scrubbed_unless_passed(dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    work, secret_dir = dirs
    monkeypatch.setenv("BRAID_TEST_SECRET", "leaky")
    sb = SeatbeltSandbox()
    pol = policy(work, secret_dir)
    hidden = await sb.run("echo \"[$BRAID_TEST_SECRET]\"; echo $TMPDIR", cwd=work, policy=pol)
    assert hidden.output.splitlines() == ["[]", str(work / ".tmp")]
    shown = await sb.run("echo \"[$BRAID_TEST_SECRET]\"", cwd=work, policy=pol, env={"BRAID_TEST_SECRET": "given"})
    assert shown.output.strip() == "[given]"


async def test_timeout_kills_the_process_group_grandchild_included(dirs: tuple[Path, Path]) -> None:
    work, secret_dir = dirs
    code = "import subprocess; p = subprocess.Popen(['sleep', '300']); print(p.pid, flush=True); p.wait()"
    result = await SeatbeltSandbox().run(f"{PY} -c \"{code}\"", cwd=work, policy=policy(work, secret_dir), timeout_s=1.5)
    assert result.timed_out and result.duration_s < 10
    grandchild = int(result.output.split()[0])
    for _ in range(20):  # reaped by launchd; give it a moment to vanish
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"grandchild {grandchild} outlived the timeout")


async def test_pytest_from_a_fixture_venv_runs_under_toolchain_paths(fixture_venv: Path, dirs: tuple[Path, Path]) -> None:
    work, secret_dir = dirs
    pol = SandboxPolicy(writable=(work, work / ".tmp"), readable=toolchain_paths(fixture_venv), denied_read=(secret_dir, Path.home()))
    result = await SeatbeltSandbox().run(f"{fixture_venv}/bin/python -m pytest --version", cwd=work, policy=pol)
    assert result.exit_code == 0 and "pytest" in result.output


async def test_output_is_tail_trimmed_with_a_head_marker(tmp_path: Path) -> None:
    pol = SandboxPolicy(writable=(tmp_path,), denied_read=())
    result = await LocalSandbox(max_output_bytes=100).run("yes abcdefghi | head -c 1000", cwd=tmp_path, policy=pol)
    assert result.output.startswith("[… trimmed 900 bytes …]\n") and len(result.output) == len("[… trimmed 900 bytes …]\n") + 100


def test_profile_escapes_quotes_and_resolves_paths(tmp_path: Path) -> None:
    odd = tmp_path / 'a"b'
    odd.mkdir()
    text = render_profile(SandboxPolicy(writable=(odd,), denied_read=()))
    assert 'a\\"b' in text and str(tmp_path.resolve()) in text
