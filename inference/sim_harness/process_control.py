from __future__ import annotations

from pathlib import Path
from time import time
import json
import shlex
import subprocess

from .config import HarnessConfig, ServerConfig, SlotConfig


def run_checked(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(detail) from exc


def run_best_effort(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, cwd=cwd, env=env)


def tmux_has_session(session_name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0


def tmux_kill_session(session_name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        text=True,
        capture_output=True,
        check=False,
    )


def tmux_capture_pane(session_name: str, lines: int = 200) -> str:
    proc = run_best_effort(["tmux", "capture-pane", "-pt", session_name, "-S", f"-{max(1, lines)}"])
    if proc.returncode != 0:
        return ""
    return proc.stdout


def tmux_send_keys(session_name: str, text: str, *, enter: bool = True) -> None:
    args = ["tmux", "send-keys", "-t", session_name, text]
    if enter:
        args.append("C-m")
    run_checked(args)


def inference_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_slot_shell_command(harness: HarnessConfig, slot: SlotConfig) -> str:
    runas = str(Path(harness.runtime.runas_path))
    root = inference_root()
    python_path = root / ".venv" / "bin" / "python"
    config_path = Path(harness.source_path or (root / "sim_harness.toml")).resolve()
    inner = (
        f"cd {shlex.quote(str(root))} && "
        f"exec {shlex.quote(str(python_path))} -m sim_harness.slot_agent "
        f"--config {shlex.quote(str(config_path))} --slot {shlex.quote(slot.name)}"
    )
    command = [
        "sudo",
        "-n",
        runas,
        slot.user,
        "bash",
        "-lc",
        inner,
    ]
    return shlex.join(command)


def tmux_new_detached_session(session_name: str, shell_command: str) -> None:
    run_checked(["tmux", "new-session", "-d", "-s", session_name, shell_command])


def check_sudo_runas_access(harness: HarnessConfig, slot: SlotConfig) -> None:
    runas = str(Path(harness.runtime.runas_path))
    run_checked(["sudo", "-n", runas, slot.user, "true"])


def launch_slot_session(harness: HarnessConfig, slot: SlotConfig) -> str:
    check_sudo_runas_access(harness, slot)
    command = build_slot_shell_command(harness, slot)
    if tmux_has_session(slot.tmux_session):
        tmux_kill_session(slot.tmux_session)
    tmux_new_detached_session(slot.tmux_session, command)
    return command


def cleanup_slot_processes(harness: HarnessConfig, slot: SlotConfig) -> dict:
    runas = str(Path(harness.runtime.runas_path))
    killed = []
    exact_names = [
        "gamescope",
        "cs2",
        "reaper",
        "srt-bwrap",
        "pv-adverb",
        "steam",
        "steamwebhelper",
        "steam-runtime-launcher-service",
    ]
    for name in exact_names:
        proc = run_best_effort(["sudo", "-n", runas, slot.user, "pkill", "-9", "-x", name])
        if proc.returncode == 0:
            killed.append(name)

    shell_script = 'rm -f "${TMPDIR:-/tmp}"/source_engine_*.lock'
    run_best_effort(["sudo", "-n", runas, slot.user, "bash", "-lc", shell_script])

    return {
        "slot": slot.name,
        "user": slot.user,
        "tmux_session": slot.tmux_session,
        "killed": killed,
        "timestamp": time(),
    }


def cleanup_global_source_locks(harness: HarnessConfig) -> list[str]:
    runas = str(Path(harness.runtime.runas_path))
    proc = run_best_effort(
        [
            "sudo",
            "-n",
            runas,
            "root",
            "bash",
            "-lc",
            'shopt -s nullglob; for f in /tmp/source_engine_*.lock; do echo "$f"; rm -f "$f"; done',
        ]
    )
    if proc.returncode != 0:
        return []
    return [line for line in (proc.stdout or "").splitlines() if line.strip()]


def read_text_via_runas(harness: HarnessConfig, user: str, path: str | Path) -> str:
    runas = str(Path(harness.runtime.runas_path))
    proc = run_checked(
        ["sudo", "-n", runas, user, "cat", str(path)],
    )
    return proc.stdout


def read_json_via_runas(harness: HarnessConfig, user: str, path: str | Path) -> dict:
    return json.loads(read_text_via_runas(harness, user, path))


def launch_server_session(harness: HarnessConfig) -> list[str]:
    if not harness.server.enabled:
        raise RuntimeError("server integration is disabled in config")
    root = inference_root().parent
    command = list(harness.server.start_command)
    run_checked(command, cwd=root)
    return command


def stop_server_session(harness: HarnessConfig) -> None:
    tmux_kill_session(harness.server.tmux_session)


def server_session_running(server: ServerConfig) -> bool:
    return tmux_has_session(server.tmux_session)
