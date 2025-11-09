import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

_PLUGINS_ENV = "DRNB_PLUGINS"
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
            root = Path(os.path.expanduser(os.path.expandvars(env_root)))
        else:
            root = Path(__file__).resolve().parent
        self.root = Path(root)
        self._by_method: dict[str, PluginSpec] = {}
        self._load()

    def _load(self) -> None:
        cfg = self.root / "plugins.toml"
        if cfg.exists():
            data = tomllib.loads(cfg.read_text(encoding="utf-8"))
            for m, entry in (data.get("plugins") or {}).items():
                folder = entry.get("folder", m)
                runner = entry.get("runner")
                key = m.lower()
                if key in self._by_method:
                    raise ValueError(f"Duplicate plugin registration for '{m}'")
                self._by_method[key] = PluginSpec(
                    method=m,
                    plugin_dir=(self.root / folder).resolve(),
                    runner=shlex.split(runner) if runner else None,
                )
        else:
            # fallback: autodiscover folders that contain the runner file
            for p in self.root.glob("*/drnb-plugin-run.py"):
                m = p.parent.name
                key = m.lower()
                if key in self._by_method:
                    raise ValueError(f"Duplicate plugin registration for '{m}'")
                self._by_method[key] = PluginSpec(
                    method=m, plugin_dir=p.parent.resolve()
                )

    def lookup(self, method: str) -> PluginSpec | None:
        return self._by_method.get(method.lower())


def plugins_enabled() -> bool:
    return os.environ.get(_PLUGINS_ENV, "1").lower() not in ("", "0", "false")


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
