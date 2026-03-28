#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo $0 <user> <command> [args...]"
  echo "Example: sudo $0 steam1 steam -tenfoot"
  exit 1
fi

TARGET_USER="$1"
shift

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
RUNTIME_DIR="/run/user/${TARGET_UID}"
BUS_PATH="${RUNTIME_DIR}/bus"

if [[ -z "${TARGET_HOME}" ]]; then
  echo "Could not determine home for user: ${TARGET_USER}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}"
chown "${TARGET_UID}:${TARGET_GID}" "${RUNTIME_DIR}"
chmod 700 "${RUNTIME_DIR}"

# Keep user runtime/session infrastructure alive across logouts.
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "${TARGET_USER}" >/dev/null 2>&1 || true
fi

# Create common user dirs if missing.
install -d -m 700 -o "${TARGET_UID}" -g "${TARGET_GID}" \
  "${TARGET_HOME}/.local" \
  "${TARGET_HOME}/.local/share" \
  "${TARGET_HOME}/.config" \
  "${TARGET_HOME}/.cache"

# Try to ensure a DBus session bus exists.
if [[ ! -S "${BUS_PATH}" ]]; then
  # First try to start a user systemd manager if available.
  if command -v runuser >/dev/null 2>&1; then
    runuser -u "${TARGET_USER}" -- env XDG_RUNTIME_DIR="${RUNTIME_DIR}" systemctl --user daemon-reexec >/dev/null 2>&1 || true
  fi

  # If still no bus, start a private dbus-daemon on the standard path.
  if [[ ! -S "${BUS_PATH}" ]]; then
    rm -f "${BUS_PATH}"
    runuser -u "${TARGET_USER}" -- \
      env HOME="${TARGET_HOME}" XDG_RUNTIME_DIR="${RUNTIME_DIR}" \
      dbus-daemon --session --address="unix:path=${BUS_PATH}" --fork --nopidfile >/dev/null 2>&1 || true
  fi
fi

export HOME="${TARGET_HOME}"
export USER="${TARGET_USER}"
export LOGNAME="${TARGET_USER}"
export XDG_RUNTIME_DIR="${RUNTIME_DIR}"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${BUS_PATH}"

# NVIDIA/gamescope env
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export GBM_BACKEND=nvidia-drm
export __NV_PRIME_RENDER_OFFLOAD=1

# Helpful defaults for Steam in headless sessions
export SDL_VIDEO_X11_DGAMOUSE=0
export STEAM_FORCE_DESKTOPUI_SCALING=1

exec runuser -u "${TARGET_USER}" -- env \
  HOME="${HOME}" \
  USER="${USER}" \
  LOGNAME="${LOGNAME}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
  DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS}" \
  __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME}" \
  GBM_BACKEND="${GBM_BACKEND}" \
  __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD}" \
  SDL_VIDEO_X11_DGAMOUSE="${SDL_VIDEO_X11_DGAMOUSE}" \
  STEAM_FORCE_DESKTOPUI_SCALING="${STEAM_FORCE_DESKTOPUI_SCALING}" \
  "$@"
