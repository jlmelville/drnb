#!/usr/bin/env bash

# Installs the SDK, core drnb package, and (optionally) every plugin workspace.
# Plugin installs are best-effort so a single failure will not abort the script.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"

echo "[drnb-install] Installing drnb-plugin-sdk from $ROOT_DIR/drnb-plugin-sdk"
"$UV_BIN" pip install -e "$ROOT_DIR/drnb-plugin-sdk"

echo "[drnb-install] Installing drnb core package from $ROOT_DIR"
"$UV_BIN" pip install -e "$ROOT_DIR"

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
    if ! "$UV_BIN" pip install -e "$plugin_dir"; then
      echo "[drnb-install] !! Failed to install plugins/$plugin_name (continuing)" >&2
    fi
  done
else
  echo "[drnb-install] No plugins directory found at $PLUGIN_ROOT; skipping plugin installs"
fi

echo "[drnb-install] Done"
