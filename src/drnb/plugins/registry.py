import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

_PLUGINS_ROOT_ENV = "DRNB_PLUGINS_ROOT"
_DEFAULT_PLUGINS_ROOT = "plugins"


@dataclass
class PluginSpec:
    method: str
    plugin_dir: Path
    runner: list[str] | None = (
        None  # e.g., ["uv", "run", "-q", "--python", "3.10", "drnb-plugin-run.py"]
    )


class Registry:
    def __init__(self) -> None:
        env_root = os.environ.get(_PLUGINS_ROOT_ENV)
        if env_root:
            root = Path(os.path.expandvars(env_root)).expanduser()
        else:
            # default to the plugins folder in the repo root
            repo_root = Path(__file__).resolve().parents[3]
            default_root = repo_root / _DEFAULT_PLUGINS_ROOT
            if default_root.exists():
                root = default_root
            else:
                root = Path(__file__).resolve().parent
        self.root = Path(root)
        self._by_method: dict[str, PluginSpec] = {}
        self._load()

    def _load(self) -> None:
        cfg = self.root / "plugins.toml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"Plugin registry missing {cfg}; create plugins.toml to register plugins"
            )

        with cfg.open("rb") as f:
            data = tomllib.load(f)

        plugins_table = data.get("plugins") or {}
        if not plugins_table:
            raise ValueError(f"No plugins configured in {cfg}")

        for m, entry in plugins_table.items():
            key = m.lower()
            if key in self._by_method:
                raise ValueError(f"Duplicate plugin registration for '{m}'")

            plugin_dir = (self.root / entry.get("folder", m)).resolve()
            if not plugin_dir.exists():
                raise FileNotFoundError(
                    f"Plugin folder for '{m}' not found at {plugin_dir}"
                )
            if not plugin_dir.is_dir():
                raise NotADirectoryError(
                    f"Plugin path for '{m}' is not a directory: {plugin_dir}"
                )

            self._by_method[key] = PluginSpec(
                method=m,
                plugin_dir=plugin_dir,
                runner=shlex.split(entry.get("runner"))
                if entry.get("runner")
                else None,
            )

    def lookup(self, method: str) -> PluginSpec | None:
        return self._by_method.get(method.lower())


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
