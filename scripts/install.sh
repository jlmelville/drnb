#!/usr/bin/env bash

# Installs the SDKs, core drnb package, and (optionally) every plugin workspace
# (both embedder plugins and NN plugins). Plugin installs are best-effort so a
# single failure will not abort the script.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
FRESH=0
REINSTALL_SDK=0
REINSTALL_ALL=0
REINSTALL_PLUGINS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/install.sh [--fresh] [--reinstall-sdk|-s] [--reinstall-all|-a] [--reinstall|-r <name>...]

Options:
  --fresh, -f           Delete each project's .venv before running `uv sync`.
  --reinstall-sdk, -s   Pass `--reinstall-package` flags so core/plugins pick up SDK
                        changes without bumping versions. Applies to both embedder
                        and NN SDKs. When the 3.10 SDK is present, it will also be
                        reinstalled.
  --reinstall-all, -a   Reinstall all plugins (embedder and NN) to pick up SDK changes
                        without bumping versions.
  --reinstall <name>, -r <name>
                        Reinstall a specific plugin by name (e.g., topometry). Can
                        be repeated to target multiple plugins. Applies to both
                        embedder plugins (plugins/<name>) and NN plugins
                        (nn-plugins/<name>). SDK reinstall is not affected when a
                        name is supplied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh|-f)
      FRESH=1
      shift
      ;;
    --reinstall-sdk|-s)
      REINSTALL_SDK=1
      shift
      ;;
    --reinstall-all|-a)
      REINSTALL_ALL=1
      shift
      ;;
    --reinstall|-r)
      if [[ -n "${2-}" && "${2#-}" == "${2}" ]]; then
        REINSTALL_PLUGINS+=("$2")
        shift 2
      else
        echo "--reinstall|-r requires a plugin name" >&2
        usage
        exit 1
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

declare -A REQUESTED_PLUGINS=()
if [[ ${#REINSTALL_PLUGINS[@]} -gt 0 ]]; then
  for name in "${REINSTALL_PLUGINS[@]}"; do
    REQUESTED_PLUGINS["$name"]=0
  done
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "[drnb-install] uv binary not found (looked for '$UV_BIN'); set UV=/path/to/uv or install uv." >&2
  exit 1
fi

sync_dir() {
  local dir="$1"
  local pkg="${2:-}"
  local reinstall="${3:-0}"
  local extra_args=()
  if [[ $FRESH -eq 1 && -d "$dir/.venv" ]]; then
    echo "[drnb-install] Removing existing virtualenv at $dir/.venv"
    rm -rf "$dir/.venv"
  fi
  if [[ $reinstall -eq 1 && -n "$pkg" ]]; then
    for p in $pkg; do
      extra_args+=(--reinstall-package "$p")
    done
    (cd "$dir" && "$UV_BIN" sync "${extra_args[@]}")
  else
    (cd "$dir" && "$UV_BIN" sync)
  fi
}

SDK_ROOT="$ROOT_DIR/plugin-sdks"
SDK_MAIN="drnb-plugin-sdk-312"
SDK_ALT="drnb-plugin-sdk-310"
NN_SDK_ROOT="$ROOT_DIR/nn-plugin-sdks"
NN_SDK_MAIN="drnb-nn-plugin-sdk-312"

echo "[drnb-install] Installing drnb-plugin-sdk-312 from $SDK_ROOT/drnb-plugin-sdk-312"
sync_dir "$SDK_ROOT/drnb-plugin-sdk-312" "drnb-plugin-sdk-312" "$REINSTALL_SDK"

if [[ -d "$SDK_ROOT/drnb-plugin-sdk-310" ]]; then
  echo "[drnb-install] Installing drnb-plugin-sdk-310 from $SDK_ROOT/drnb-plugin-sdk-310"
  sync_dir "$SDK_ROOT/drnb-plugin-sdk-310" "drnb-plugin-sdk-310" "$REINSTALL_SDK"
fi

if [[ -d "$NN_SDK_ROOT/$NN_SDK_MAIN" ]]; then
  echo "[drnb-install] Installing $NN_SDK_MAIN from $NN_SDK_ROOT/$NN_SDK_MAIN"
  sync_dir "$NN_SDK_ROOT/$NN_SDK_MAIN" "$NN_SDK_MAIN" "$REINSTALL_SDK"
fi

echo "[drnb-install] Installing drnb core package from $ROOT_DIR"
sync_dir "$ROOT_DIR" "$SDK_MAIN $NN_SDK_MAIN" "$REINSTALL_SDK"

PLUGIN_ROOT="$ROOT_DIR/plugins"
if [[ -d "$PLUGIN_ROOT" ]]; then
  echo "[drnb-install] Installing plugins under $PLUGIN_ROOT (best effort)"
  for plugin_dir in "$PLUGIN_ROOT"/*; do
    [[ -d "$plugin_dir" ]] || continue
    if [[ ! -f "$plugin_dir/pyproject.toml" ]]; then
      continue
    fi
    plugin_name="${plugin_dir##*/}"
    reinstall_plugin=0
    if [[ $REINSTALL_ALL -eq 1 ]]; then
      reinstall_plugin=1
      if [[ -n "${REQUESTED_PLUGINS[$plugin_name]+_}" ]]; then
        REQUESTED_PLUGINS["$plugin_name"]=1
      fi
    elif [[ ${#REINSTALL_PLUGINS[@]} -gt 0 ]]; then
      if [[ -n "${REQUESTED_PLUGINS[$plugin_name]+_}" ]]; then
        REQUESTED_PLUGINS["$plugin_name"]=1
        reinstall_plugin=1
      else
        continue
      fi
    fi

    pkg_flag="$SDK_MAIN"
    if grep -q "drnb-plugin-sdk-310" "$plugin_dir/pyproject.toml"; then
      pkg_flag="$SDK_ALT"
    fi

    echo "[drnb-install] -> plugins/$plugin_name"
    if ! sync_dir "$plugin_dir" "$pkg_flag" "$reinstall_plugin"; then
      echo "[drnb-install] !! Failed to install plugins/$plugin_name (continuing)" >&2
    fi
  done
else
  echo "[drnb-install] No plugins directory found at $PLUGIN_ROOT; skipping plugin installs"
fi

NN_PLUGIN_ROOT="$ROOT_DIR/nn-plugins"
if [[ -d "$NN_PLUGIN_ROOT" ]]; then
  echo "[drnb-install] Installing NN plugins under $NN_PLUGIN_ROOT (best effort)"
  for plugin_dir in "$NN_PLUGIN_ROOT"/*; do
    [[ -d "$plugin_dir" ]] || continue
    if [[ ! -f "$plugin_dir/pyproject.toml" ]]; then
      continue
    fi
    plugin_name="${plugin_dir##*/}"
    reinstall_plugin=0
    if [[ $REINSTALL_ALL -eq 1 ]]; then
      reinstall_plugin=1
      if [[ -n "${REQUESTED_PLUGINS[$plugin_name]+_}" ]]; then
        REQUESTED_PLUGINS["$plugin_name"]=1
      fi
    elif [[ ${#REINSTALL_PLUGINS[@]} -gt 0 ]]; then
      if [[ -n "${REQUESTED_PLUGINS[$plugin_name]+_}" ]]; then
        REQUESTED_PLUGINS["$plugin_name"]=1
        reinstall_plugin=1
      else
        continue
      fi
    fi

    echo "[drnb-install] -> nn-plugins/$plugin_name"
    if ! sync_dir "$plugin_dir" "$NN_SDK_MAIN" "$reinstall_plugin"; then
      echo "[drnb-install] !! Failed to install nn-plugins/$plugin_name (continuing)" >&2
    fi
  done
else
  echo "[drnb-install] No NN plugins directory found at $NN_PLUGIN_ROOT; skipping NN plugin installs"
fi

if [[ ${#REINSTALL_PLUGINS[@]} -gt 0 ]]; then
  for name in "${!REQUESTED_PLUGINS[@]}"; do
    if [[ ${REQUESTED_PLUGINS[$name]} -eq 0 ]]; then
      echo "[drnb-install] !! Requested plugin '$name' not found under plugins/ or nn-plugins/" >&2
    fi
  done
fi

echo "[drnb-install] Done"
