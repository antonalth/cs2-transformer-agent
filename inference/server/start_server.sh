#!/usr/bin/env bash
set -euo pipefail

RUNAS="${RUNAS:-/usr/local/bin/runas}"
SERVER_USER="${SERVER_USER:-server}"
SERVER_HOME="${SERVER_HOME:-/home/server}"
CS2_ROOT_DEFAULT="${SERVER_HOME}/steam-lib/steamapps/common/Counter-Strike Global Offensive"
CS2_ROOT="${CS2_ROOT:-$CS2_ROOT_DEFAULT}"
GAME_DIR="${GAME_DIR:-${CS2_ROOT}/game}"
TMUX_SESSION="${TMUX_SESSION:-cs2-ds}"
MAP="${MAP:-de_dust2}"
PORT="${PORT:-27015}"
GAME_TYPE="${GAME_TYPE:-0}"
GAME_MODE="${GAME_MODE:-1}"
HOSTNAME_VALUE="${HOSTNAME_VALUE:-cs2-sim-harness}"
GSLT="${GSLT:-}"
SV_LAN="${SV_LAN:-1}"
INSECURE="${INSECURE:-1}"
STEAMRT_DIR="${STEAMRT_DIR:-${SERVER_HOME}/steamrt}"
CS2_BIN="${GAME_DIR}/bin/linuxsteamrt64/cs2"
SERVER_LAUNCH_MODE="${SERVER_LAUNCH_MODE:-default}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TMUX_SESSION}"
fi

GSLT_ARG=""
if [[ -n "${GSLT}" ]]; then
  GSLT_ARG="+sv_setsteamaccount ${GSLT}"
fi

LAN_ARG=""
if [[ "${SV_LAN}" == "1" ]]; then
  LAN_ARG="+sv_lan 1"
fi

INSECURE_ARG=""
if [[ "${INSECURE}" == "1" ]]; then
  INSECURE_ARG="-insecure"
fi

LAUNCH_CMD="./cs2.sh"
case "${SERVER_LAUNCH_MODE}" in
  default)
    ;;
  steamrt-wrapper)
    if sudo -n "${RUNAS}" "${SERVER_USER}" test -x "${STEAMRT_DIR}/run"; then
      LAUNCH_CMD="${STEAMRT_DIR}/run -- ./cs2.sh"
    else
      echo "SteamRT wrapper mode requested but ${STEAMRT_DIR}/run is missing" >&2
      exit 1
    fi
    ;;
  steamrt-direct)
    if sudo -n "${RUNAS}" "${SERVER_USER}" test -x "${STEAMRT_DIR}/run"; then
      LAUNCH_CMD="${STEAMRT_DIR}/run -- \"${CS2_BIN}\" --graphics-provider \"\" --"
    else
      echo "SteamRT direct mode requested but ${STEAMRT_DIR}/run is missing" >&2
      exit 1
    fi
    ;;
  *)
    echo "Unknown SERVER_LAUNCH_MODE=${SERVER_LAUNCH_MODE}" >&2
    exit 1
    ;;
esac

INNER_CMD=$(
  cat <<EOF
cd "${GAME_DIR}" && exec ${LAUNCH_CMD} -dedicated -usercon ${INSECURE_ARG} \
  +hostname "${HOSTNAME_VALUE}" \
  ${LAN_ARG} \
  +map "${MAP}" \
  +ip 0.0.0.0 \
  -port "${PORT}" \
  +game_type "${GAME_TYPE}" \
  +game_mode "${GAME_MODE}" \
  ${GSLT_ARG}
EOF
)

TMUX_CMD=$(
  printf 'sudo -n %q %q bash -lc %q' \
    "${RUNAS}" \
    "${SERVER_USER}" \
    "${INNER_CMD}"
)

tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_CMD}"

echo "Started dedicated server in tmux session ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
