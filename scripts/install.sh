#!/usr/bin/env bash

# Installs the SDK, core drnb package, and (optionally) every plugin workspace.
# Plugin installs are best-effort so a single failure will not abort the script.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
FRESH=0
REINSTALL_SDK=0

usage() {
  cat <<'EOF'
Usage: ./scripts/install.sh [--fresh] [--reinstall-sdk]

Options:
  --fresh, -f          Delete each project's .venv before running `uv sync`.
  --reinstall-sdk, -r  Pass `--reinstall-package drnb-plugin-sdk-312` to `uv sync`
                       so core/plugins pick up SDK changes without bumping the
                       version. When the 3.10 SDK is present, it will also be
                       reinstalled.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh|-f)
      FRESH=1
      shift
      ;;
    --reinstall-sdk|-r)
      REINSTALL_SDK=1
      shift
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

sync_dir() {
  local dir="$1"
  local pkg="${2:-}"
  if [[ $FRESH -eq 1 && -d "$dir/.venv" ]]; then
    echo "[drnb-install] Removing existing virtualenv at $dir/.venv"
    rm -rf "$dir/.venv"
  fi
  if [[ $REINSTALL_SDK -eq 1 && -n "$pkg" ]]; then
    (cd "$dir" && "$UV_BIN" sync --reinstall-package "$pkg")
  else
    (cd "$dir" && "$UV_BIN" sync)
  fi
}

SDK_ROOT="$ROOT_DIR/plugin-sdks"

echo "[drnb-install] Installing drnb-plugin-sdk-312 from $SDK_ROOT/drnb-plugin-sdk-312"
sync_dir "$SDK_ROOT/drnb-plugin-sdk-312" "drnb-plugin-sdk-312"

if [[ -d "$SDK_ROOT/drnb-plugin-sdk-310" ]]; then
  echo "[drnb-install] Installing drnb-plugin-sdk-310 from $SDK_ROOT/drnb-plugin-sdk-310"
  sync_dir "$SDK_ROOT/drnb-plugin-sdk-310" "drnb-plugin-sdk-310"
fi

echo "[drnb-install] Installing drnb core package from $ROOT_DIR"
sync_dir "$ROOT_DIR"

PLUGIN_ROOT="$ROOT_DIR/plugins"
if [[ -d "$PLUGIN_ROOT" ]]; then
  echo "[drnb-install] Installing plugins under $PLUGIN_ROOT (best effort)"
  for plugin_dir in "$PLUGIN_ROOT"/*; do
    [[ -d "$plugin_dir" ]] || continue
    if [[ ! -f "$plugin_dir/pyproject.toml" ]]; then
      continue
    fi
    plugin_name="${plugin_dir##*/}"
    echo "[drnb-install] -> plugins/$plugin_name"
    if ! sync_dir "$plugin_dir"; then
      echo "[drnb-install] !! Failed to install plugins/$plugin_name (continuing)" >&2
    fi
  done
else
  echo "[drnb-install] No plugins directory found at $PLUGIN_ROOT; skipping plugin installs"
fi

echo "[drnb-install] Done"
