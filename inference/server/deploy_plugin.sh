#!/usr/bin/env bash
set -euo pipefail

RUNAS="${RUNAS:-/usr/local/bin/runas}"
SERVER_USER="${SERVER_USER:-server}"
SERVER_HOME="${SERVER_HOME:-/home/server}"
CS2_ROOT_DEFAULT="${SERVER_HOME}/steam-lib/steamapps/common/Counter-Strike Global Offensive"
CS2_ROOT="${CS2_ROOT:-$CS2_ROOT_DEFAULT}"
PLUGIN_DEST="${PLUGIN_DEST:-${CS2_ROOT}/game/csgo/addons/counterstrikesharp/plugins/Cs2SimHarness}"

BUILD_DIR=""

usage() {
  cat <<'EOF'
Usage:
  deploy_plugin.sh --build-dir <path/to/net8.0/output>

Expected files in build dir:
  - Cs2SimHarness.dll
  - Cs2SimHarness.deps.json
Optional:
  - Cs2SimHarness.pdb
  - extra dependency DLLs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
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

if [[ -z "${BUILD_DIR}" || ! -d "${BUILD_DIR}" ]]; then
  echo "Build dir not found: ${BUILD_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BUILD_DIR}/Cs2SimHarness.dll" ]]; then
  echo "Missing Cs2SimHarness.dll in ${BUILD_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BUILD_DIR}/Cs2SimHarness.deps.json" ]]; then
  echo "Missing Cs2SimHarness.deps.json in ${BUILD_DIR}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying plugin to ${PLUGIN_DEST}..."
sudo -n "${RUNAS}" "${SERVER_USER}" bash -lc "
  set -euo pipefail
  mkdir -p \"${PLUGIN_DEST}\"
  cp \"${BUILD_DIR}\"/Cs2SimHarness.* \"${PLUGIN_DEST}/\"
  find \"${BUILD_DIR}\" -maxdepth 1 -type f -name '*.dll' ! -name 'Cs2SimHarness.dll' -exec cp {} \"${PLUGIN_DEST}/\" \;
  if [[ ! -f \"${PLUGIN_DEST}/cs2-sim-harness.json\" ]]; then
    cp \"${SCRIPT_DIR}/runtime/plugins/Cs2SimHarness/cs2-sim-harness.example.json\" \
      \"${PLUGIN_DEST}/cs2-sim-harness.json\"
  fi
"

echo "Plugin deployed."
