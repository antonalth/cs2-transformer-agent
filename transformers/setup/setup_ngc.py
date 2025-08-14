#!/usr/bin/env python3
"""
Installs Docker Engine, Docker Buildx/Compose plugins, and NVIDIA Container Toolkit on Ubuntu.
- Adds Docker APT repo + GPG key
- Installs docker-ce, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin
- Creates 'docker' group and adds the current user to it
- Installs recommended NVIDIA drivers (ubuntu-drivers install)
- Adds NVIDIA Container Toolkit repository + keyring and installs pinned versions
- Asks for reboot after completion

All package operations are non-interactive; no 'y' prompts required.
"""

import os
import sys
import shlex
import subprocess
import getpass
from pathlib import Path

# ---------- Configuration ----------
NVIDIA_CONTAINER_TOOLKIT_VERSION = "1.17.8-1"  # matches your request
DOCKER_GPG_URL = "https://download.docker.com/linux/ubuntu/gpg"
DOCKER_LIST_PATH = "/etc/apt/sources.list.d/docker.list"
DOCKER_KEYRING = "/etc/apt/keyrings/docker.asc"
NVIDIA_GPG_KEY_URL = "https://nvidia.github.io/libnvidia-container/gpgkey"
NVIDIA_LIST_RAW_URL = "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
NVIDIA_KEYRING = "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

# ---------- Helpers ----------
def is_root() -> bool:
    return os.geteuid() == 0

def sudo_prefix() -> list[str]:
    return [] if is_root() else ["sudo", "-n"]

def run_cmd(cmd: str | list[str], *, check: bool = True, env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """
    Run a shell command with optional sudo, raising on failure with helpful output.
    """
    if isinstance(cmd, str):
        args = cmd if is_root() else " ".join(map(shlex.quote, sudo_prefix() + [cmd]))
        # If cmd is str, run through shell to preserve pipes/redirects when needed.
        result = subprocess.run(args, shell=True, capture_output=True, text=True, env=env, cwd=cwd)
    else:
        args = sudo_prefix() + cmd
        result = subprocess.run(args, capture_output=True, text=True, env=env, cwd=cwd)

    if check and result.returncode != 0:
        sys.stderr.write(f"\n[ERROR] Command failed ({result.returncode}): {cmd}\n")
        if result.stdout:
            sys.stderr.write(f"stdout:\n{result.stdout}\n")
        if result.stderr:
            sys.stderr.write(f"stderr:\n{result.stderr}\n")
        raise SystemExit(result.returncode)
    return result

def require_ubuntu():
    try:
        data = Path("/etc/os-release").read_text()
    except Exception:
        print("Could not read /etc/os-release; this script supports Ubuntu only.", file=sys.stderr)
        raise SystemExit(1)
    if "ID=ubuntu" not in data and 'ID_LIKE=ubuntu' not in data:
        print("This script is intended for Ubuntu (or Ubuntu-like). Aborting.", file=sys.stderr)
        raise SystemExit(1)

def get_ubuntu_codename() -> str:
    codename = None
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("UBUNTU_CODENAME=") or line.startswith("VERSION_CODENAME="):
                    codename = line.strip().split("=", 1)[1].strip().strip('"')
                    if codename:
                        break
    except Exception:
        pass
    if not codename:
        # Fallback via lsb_release if available
        try:
            out = subprocess.check_output(["lsb_release", "-cs"], text=True).strip()
            codename = out
        except Exception:
            pass
    if not codename:
        print("Unable to determine Ubuntu codename.", file=sys.stderr)
        raise SystemExit(1)
    return codename

def get_arch() -> str:
    try:
        out = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
        return out
    except Exception:
        print("Unable to determine dpkg architecture.", file=sys.stderr)
        raise SystemExit(1)

def ensure_group(group: str):
    # Check if group exists; if not, create it
    res = subprocess.run(["getent", "group", group], capture_output=True, text=True)
    if res.returncode != 0:
        run_cmd(["groupadd", group])

def add_user_to_group(user: str, group: str):
    run_cmd(["usermod", "-aG", group, user])

def write_file_atomic(path: str, content: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
    # Move into place with sudo if needed
    if is_root():
        os.replace(tmp_path, path)
    else:
        # Move with sudo mv
        run_cmd(["mv", tmp_path, path])

def print_step(title: str):
    print(f"\n=== {title} ===")

def confirm_reboot():
    try:
        ans = input("\nSetup complete. A reboot is recommended to apply group and driver changes.\nReboot now? [y/N]: ").strip().lower()
        if ans == "y":
            print("Rebooting...")
            run_cmd(["reboot"], check=False)
        else:
            print("Okay! Please reboot later to finalize the installation.")
    except EOFError:
        # Non-interactive: just print a reminder
        print("\nNon-interactive session detected. Please reboot your system to finalize the installation.")

# ---------- Main ----------
def main():
    require_ubuntu()
    user = getpass.getuser()
    arch = get_arch()
    codename = get_ubuntu_codename()

    # Ensure non-interactive apt
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    env["TZ"] = env.get("TZ", "Etc/UTC")

    print_step("Updating APT and installing prerequisites")
    run_cmd(["apt-get", "update"], env=env)
    run_cmd(["apt-get", "install", "-y", "ca-certificates", "curl", "gnupg", "lsb-release"], env=env)

    print_step("Preparing keyrings directory")
    run_cmd(["install", "-m", "0755", "-d", "/etc/apt/keyrings"], env=env)

    print_step("Adding Docker's official GPG key")
    run_cmd(f'curl -fsSL {shlex.quote(DOCKER_GPG_URL)} -o {shlex.quote(DOCKER_KEYRING)}', env=env)
    run_cmd(["chmod", "a+r", DOCKER_KEYRING], env=env)

    print_step("Adding Docker APT repository")
    docker_list_content = (
        f"deb [arch={arch} signed-by={DOCKER_KEYRING}] https://download.docker.com/linux/ubuntu {codename} stable\n"
    )
    # Write to temp and move into place with appropriate perms
    write_file_atomic(DOCKER_LIST_PATH, docker_list_content)

    print_step("APT update (Docker repo)")
    run_cmd(["apt-get", "update"], env=env)

    print_step("Installing Docker Engine & plugins (non-interactive)")
    run_cmd([
        "apt-get", "install", "-y",
        "docker-ce", "docker-ce-cli", "containerd.io",
        "docker-buildx-plugin", "docker-compose-plugin"
    ], env=env)

    print_step("Creating 'docker' group (if missing) and adding current user")
    ensure_group("docker")
    add_user_to_group(user, "docker")

    print_step("Installing recommended NVIDIA drivers (this may take a while)")
    # ubuntu-drivers install is non-interactive and installs recommended drivers
    run_cmd(["ubuntu-drivers", "install"], env=env)

    print_step("Adding NVIDIA Container Toolkit repository & keyring")
    # Fetch NVIDIA key; dearmor into keyring
    run_cmd(f'curl -fsSL {shlex.quote(NVIDIA_GPG_KEY_URL)} | '
            f'{" ".join(sudo_prefix())} gpg --dearmor -o {shlex.quote(NVIDIA_KEYRING)}', env=env, check=True)

    # Fetch repo list, patch with signed-by option, and install to sources.list.d
    # Using a small Python-driven pipeline to avoid sed portability issues
    import urllib.request
    with urllib.request.urlopen(NVIDIA_LIST_RAW_URL) as resp:
        raw_list = resp.read().decode("utf-8")
    patched_list = raw_list.replace(
        "deb https://", f"deb [signed-by={NVIDIA_KEYRING}] https://"
    )
    write_file_atomic("/etc/apt/sources.list.d/nvidia-container-toolkit.list", patched_list)

    print_step("APT update (NVIDIA Container Toolkit repo)")
    run_cmd(["apt-get", "update"], env=env)

    print_step("Installing NVIDIA Container Toolkit (pinned versions)")
    run_cmd([
        "apt-get", "install", "-y",
        f"nvidia-container-toolkit={NVIDIA_CONTAINER_TOOLKIT_VERSION}",
        f"nvidia-container-toolkit-base={NVIDIA_CONTAINER_TOOLKIT_VERSION}",
        f"libnvidia-container-tools={NVIDIA_CONTAINER_TOOLKIT_VERSION}",
        f"libnvidia-container1={NVIDIA_CONTAINER_TOOLKIT_VERSION}",
    ], env=env)

    print_step("All steps completed successfully")
    print("\nNotes:")
    print("- You were added to the 'docker' group; this takes effect after you log out and back in, or after a reboot.")
    print("- If you installed new NVIDIA drivers, a reboot is strongly recommended.")

    confirm_reboot()

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        sys.exit(e.code)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)
