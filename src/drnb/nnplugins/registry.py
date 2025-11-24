import os
import shlex
from dataclasses import dataclass
from pathlib import Path

_NN_PLUGINS_ROOT_ENV = "DRNB_NN_PLUGINS_ROOT"
_DEFAULT_NN_PLUGINS_ROOT = "nn-plugins"


def _validate_directory(path: Path, context: str) -> Path:
    """Validate that a path exists and is a directory."""
    if not path.exists():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{context} is not a directory: {path}")
    return path


@dataclass
class NNPluginSpec:
    method: str
    plugin_dir: Path
    runner: list[str] | None = None


class NNRegistry:
    def __init__(self) -> None:
        env_root = os.environ.get(_NN_PLUGINS_ROOT_ENV)
        if env_root:
            root = Path(os.path.expandvars(env_root)).expanduser()
            _validate_directory(root, f"NN plugin root from {_NN_PLUGINS_ROOT_ENV}")
        else:
            repo_root = Path(__file__).resolve().parents[3]
            default_root = repo_root / _DEFAULT_NN_PLUGINS_ROOT
            _validate_directory(default_root, "NN plugin root")
            root = default_root
        self.root = root
        self._by_method: dict[str, NNPluginSpec] = {}
        self._load()

    def _load(self) -> None:
        cfg = self.root / "plugins.toml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"NN plugin registry missing {cfg}; create plugins.toml to register NN plugins"
            )

        import tomllib

        with cfg.open("rb") as f:
            data = tomllib.load(f)

        plugins_table = data.get("plugins") or {}
        if not plugins_table:
            raise ValueError(f"No NN plugins configured in {cfg}")

        for m, entry in plugins_table.items():
            key = m.lower()
            if key in self._by_method:
                raise ValueError(f"Duplicate NN plugin registration for '{m}'")

            plugin_dir = (self.root / entry.get("folder", m)).resolve()
            _validate_directory(plugin_dir, f"NN plugin folder for '{m}'")

            self._by_method[key] = NNPluginSpec(
                method=m,
                plugin_dir=plugin_dir,
                runner=shlex.split(entry.get("runner"))
                if entry.get("runner")
                else None,
            )

    def lookup(self, method: str) -> NNPluginSpec | None:
        return self._by_method.get(method.lower())


_registry: NNRegistry | None = None


def get_registry() -> NNRegistry:
    global _registry
    if _registry is None:
        _registry = NNRegistry()
    return _registry
