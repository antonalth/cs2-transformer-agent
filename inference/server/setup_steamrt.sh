#!/usr/bin/env bash
set -euo pipefail

RUNAS="${RUNAS:-/usr/local/bin/runas}"
SERVER_USER="${SERVER_USER:-server}"
SERVER_HOME="${SERVER_HOME:-/home/server}"
STEAMRT_DIR="${STEAMRT_DIR:-${SERVER_HOME}/steamrt}"
STEAMCMD_BIN="${STEAMCMD_BIN:-$(command -v steamcmd || true)}"
STEAMRT_APPID="${STEAMRT_APPID:-1628350}"

if [[ -z "${STEAMCMD_BIN}" ]]; then
  echo "steamcmd not found on host" >&2
  exit 1
fi

sudo -n "${RUNAS}" "${SERVER_USER}" bash -lc "
  set -euo pipefail
  mkdir -p \"${STEAMRT_DIR}\"
  \"${STEAMCMD_BIN}\" \
    +@sSteamCmdForcePlatformType linux \
    +force_install_dir \"${STEAMRT_DIR}\" \
    +login anonymous \
    +app_update ${STEAMRT_APPID} \
    +quit
  test -x \"${STEAMRT_DIR}/run\"
  echo \"SteamRT installed at ${STEAMRT_DIR}\"
"
