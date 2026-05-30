from __future__ import annotations

from importlib import metadata

UNKNOWN_VALUE = "unknown"

_PACKAGE_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
}


def _normalize_package(name: str) -> str:
    return _PACKAGE_ALIASES.get(name.lower(), name)


def get_package_version(package: str | None) -> str | None:
    if not package:
        return None
    normalized = _normalize_package(package)
    try:
        return metadata.version(normalized)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def build_version_payload(package: str) -> dict[str, str]:
    """Construct a standard version payload for plugin responses."""
    return {
        "package": package,
        "version": get_package_version(package) or UNKNOWN_VALUE,
    }
