#!/usr/bin/env bash
set -euo pipefail

RUNAS="${RUNAS:-/usr/local/bin/runas}"
SERVER_USER="${SERVER_USER:-server}"
SERVER_HOME="${SERVER_HOME:-/home/server}"
CS2_ROOT_DEFAULT="${SERVER_HOME}/steam-lib/steamapps/common/Counter-Strike Global Offensive"
CS2_ROOT="${CS2_ROOT:-$CS2_ROOT_DEFAULT}"
GAME_DIR="${GAME_DIR:-${CS2_ROOT}/game}"
MOD_DIR="${MOD_DIR:-${GAME_DIR}/csgo}"
GAMEINFO="${GAMEINFO:-${MOD_DIR}/gameinfo.gi}"

METAMOD_ARCHIVE=""
CSSHARP_ARCHIVE=""

usage() {
  cat <<'EOF'
Usage:
  install_mod_stack.sh --metamod-archive <path> --cssharp-archive <path>

Notes:
  - Use the Linux Metamod:Source archive.
  - Use the CounterStrikeSharp "with-runtime" Linux archive for first install.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metamod-archive)
      METAMOD_ARCHIVE="$2"
      shift 2
      ;;
    --cssharp-archive)
      CSSHARP_ARCHIVE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${METAMOD_ARCHIVE}" || -z "${CSSHARP_ARCHIVE}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${METAMOD_ARCHIVE}" ]]; then
  echo "Metamod archive not found: ${METAMOD_ARCHIVE}" >&2
  exit 1
fi

if [[ ! -f "${CSSHARP_ARCHIVE}" ]]; then
  echo "CounterStrikeSharp archive not found: ${CSSHARP_ARCHIVE}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSSHARP_SO="${MOD_DIR}/addons/counterstrikesharp/bin/linuxsteamrt64/counterstrikesharp.so"

echo "Ensuring server Steam bootstrap exists..."
"${SCRIPT_DIR}/bootstrap_server_user.sh"

if ! sudo -n "${RUNAS}" "${SERVER_USER}" test -f "${GAMEINFO}"; then
  echo "gameinfo.gi not found for ${SERVER_USER}: ${GAMEINFO}" >&2
  exit 1
fi

echo "Installing Metamod and CounterStrikeSharp into ${MOD_DIR}..."
extract_archive() {
  local archive_path="$1"
  local dest_dir="$2"

  case "${archive_path}" in
    *.tar.gz|*.tgz)
      tar -xf "${archive_path}" -C "${dest_dir}"
      ;;
    *.zip)
      python3 - "${archive_path}" "${dest_dir}" <<'PY'
import sys
import zipfile

archive_path, dest_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(archive_path) as zf:
    zf.extractall(dest_dir)
PY
      ;;
    *)
      echo "Unsupported archive type: ${archive_path}" >&2
      exit 1
      ;;
  esac
}

sudo -n "${RUNAS}" "${SERVER_USER}" bash -lc "
  set -euo pipefail
  $(declare -f extract_archive)
  mkdir -p \"${MOD_DIR}\"
  extract_archive \"${METAMOD_ARCHIVE}\" \"${MOD_DIR}\"
  extract_archive \"${CSSHARP_ARCHIVE}\" \"${MOD_DIR}\"
"

echo "Patching ${GAMEINFO} for Metamod..."
sudo -n "${RUNAS}" "${SERVER_USER}" python3 - "$GAMEINFO" <<'PY'
from pathlib import Path
import sys

gameinfo_path = Path(sys.argv[1])
text = gameinfo_path.read_text(encoding="utf-8")
needle = '\t\t\tGame_LowViolence\tcsgo_lv // Perfect World content override\n'
insert = '\t\t\tGame\tcsgo/addons/metamod\n'

if insert in text:
    print("Metamod search path already present.")
    raise SystemExit(0)

if needle not in text:
    raise SystemExit(f"Could not find insertion point in {gameinfo_path}")

backup = gameinfo_path.with_suffix(".gi.bak")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

text = text.replace(needle, needle + insert, 1)
gameinfo_path.write_text(text, encoding="utf-8")
print(f"Patched {gameinfo_path}")
print(f"Backup at {backup}")
PY

echo "Installing runtime config templates..."
sudo -n "${RUNAS}" root bash -lc "
  set -euo pipefail
  mkdir -p \"${MOD_DIR}/cfg\"
  cp \"${SCRIPT_DIR}/runtime/cfg/server.cfg\" \"${MOD_DIR}/cfg/server.cfg\"
  cp \"${SCRIPT_DIR}/runtime/cfg/gamemode_competitive_server.cfg\" \"${MOD_DIR}/cfg/gamemode_competitive_server.cfg\"
  mkdir -p \"${MOD_DIR}/addons/counterstrikesharp/plugins/Cs2SimHarness\"
  if [[ ! -f \"${MOD_DIR}/addons/counterstrikesharp/plugins/Cs2SimHarness/cs2-sim-harness.json\" ]]; then
    cp \"${SCRIPT_DIR}/runtime/plugins/Cs2SimHarness/cs2-sim-harness.example.json\" \
      \"${MOD_DIR}/addons/counterstrikesharp/plugins/Cs2SimHarness/cs2-sim-harness.json\"
  fi
  chown -R \"${SERVER_USER}:${SERVER_USER}\" \
    \"${MOD_DIR}/cfg/server.cfg\" \
    \"${MOD_DIR}/cfg/gamemode_competitive_server.cfg\" \
    \"${MOD_DIR}/addons/counterstrikesharp/plugins/Cs2SimHarness\"
"

if sudo -n "${RUNAS}" "${SERVER_USER}" test -f "${CSSHARP_SO}"; then
  echo "Clearing executable-stack flag on CounterStrikeSharp..."
  sudo -n "${RUNAS}" root python3 "${SCRIPT_DIR}/clear_execstack.py" --backup "${CSSHARP_SO}"
fi

echo "Install complete."
echo "Next steps:"
echo "  1. Build or deploy the Cs2SimHarness plugin DLLs"
echo "  2. Start the server and run: meta list"
