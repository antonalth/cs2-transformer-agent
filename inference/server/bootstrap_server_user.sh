#!/usr/bin/env bash
set -euo pipefail

RUNAS="${RUNAS:-/usr/local/bin/runas}"
SERVER_USER="${SERVER_USER:-server}"

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required" >&2
  exit 1
fi

echo "Bootstrapping Steam for ${SERVER_USER}..."
sudo -n "${RUNAS}" "${SERVER_USER}" bash -lc '
  set -euo pipefail
  timeout --signal=TERM --kill-after=5 180 steam -version >/dev/null 2>&1 || true
  test -f "${HOME}/.steam/sdk64/steamclient.so"
  test -f "${HOME}/.local/share/Steam/linux64/steamclient.so"
  echo "Steam bootstrap complete for ${USER}"
  echo "  sdk64: ${HOME}/.steam/sdk64/steamclient.so"
  echo "  steamclient: ${HOME}/.local/share/Steam/linux64/steamclient.so"
'
