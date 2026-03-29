from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from .config import HarnessConfig, SlotConfig


def run_checked(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(detail) from exc


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
