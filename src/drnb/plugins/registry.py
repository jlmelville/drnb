import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

_PLUGINS_ROOT_ENV = "DRNB_PLUGINS_ROOT"
_DEFAULT_PLUGINS_ROOT = "plugins"


def _validate_directory(
    path: Path, context: str, not_found_msg: str | None = None
) -> Path:
    """Validate that a path exists and is a directory, raising appropriate errors.

    Args:
        path: The path to validate
        context: Descriptive context for error messages (e.g., "from DRNB_PLUGINS_ROOT")
        not_found_msg: Optional custom message for FileNotFoundError. If None, uses
            default format: "{context} does not exist: {path}"

    Returns:
        The validated path

    Raises:
        FileNotFoundError: If the path does not exist
        NotADirectoryError: If the path exists but is not a directory
    """
    if not path.exists():
        msg = not_found_msg or f"{context} does not exist: {path}"
        raise FileNotFoundError(msg)
    if not path.is_dir():
        raise NotADirectoryError(f"{context} is not a directory: {path}")
    return path


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
            _validate_directory(root, f"Plugin root directory from {_PLUGINS_ROOT_ENV}")
        else:
            # default to the plugins folder in the repo root
            repo_root = Path(__file__).resolve().parents[3]
            default_root = repo_root / _DEFAULT_PLUGINS_ROOT
            _validate_directory(
                default_root,
                "Plugin root",
                not_found_msg=(
                    f"Plugin root directory does not exist at {default_root}. "
                    f"Set {_PLUGINS_ROOT_ENV} to point to your plugins directory."
                ),
            )
            root = default_root
        self.root = root
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
            _validate_directory(plugin_dir, f"Plugin folder for '{m}'")

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
