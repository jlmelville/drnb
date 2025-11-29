from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

from drnb.log import log

UNKNOWN_VALUE = "unknown"

_PACKAGE_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
}

# Map known embedder method names (lowercase) to the distribution that should
# report the version. Any method not listed here will fall back to module name
# or the root package for the embedder implementation.
_METHOD_PACKAGE_MAP: dict[str, str] = {
    # sklearn-based
    "pca": "scikit-learn",
    "randproj": "scikit-learn",
    "isomap": "scikit-learn",
    "sklearn-mmds": "scikit-learn",
    "sklearn-nmds": "scikit-learn",
    "skmmds": "scikit-learn",
    "sikmmds": "scikit-learn",
    "rsikmmds": "scikit-learn",
    "mrsikmmds": "scikit-learn",
    "lcmmds": "scikit-learn",
    # umap-learn based
    "umap": "umap-learn",
    "umap2": "umap-learn",
    "umapspectral": "umap-learn",
    "bgspectral": "umap-learn",
    "pacumap": "umap-learn",
    "htumap": "umap-learn",
    "htnegumap": "umap-learn",
    "negumap": "umap-learn",
    "negtumap": "umap-learn",
    "negtsne": "umap-learn",
}


def _normalize_package_name(name: str) -> str:
    return _PACKAGE_ALIASES.get(name.lower(), name)


def _metadata_version(name: str | None) -> str | None:
    if not name:
        return None
    candidate = _normalize_package_name(name)
    try:
        return metadata.version(candidate)
    except metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("Version lookup failed for %s: %s", candidate, exc)
    return None


def _module_version(name: str | None) -> str | None:
    if not name:
        return None
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        return None
    return str(version)


def _candidate_packages(embedder: Any, method_hint: str | None) -> list[str]:
    candidates: list[str] = []
    if method_hint:
        method_hint = method_hint.lower()
        mapped = _METHOD_PACKAGE_MAP.get(method_hint)
        candidates.append(mapped or method_hint)
    method_attr = getattr(embedder, "method", None)
    if method_attr:
        candidates.append(str(method_attr).lower())
    module_root = str(embedder.__module__).split(".", maxsplit=1)[0]
    candidates.append(module_root)
    if module_root == "drnb" or embedder.__module__.startswith("drnb."):
        candidates.append("drnb")
    # dedupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for cand in candidates:
        if cand not in seen:
            ordered.append(cand)
            seen.add(cand)
    return ordered


def _candidate_modules(embedder: Any) -> list[str]:
    modules = [embedder.__module__]
    root = embedder.__module__.split(".", maxsplit=1)[0]
    if root not in modules:
        modules.append(root)
    return modules


def _resolve_single_version(embedder: Any, method_hint: str | None) -> dict[str, Any]:
    package_candidates = _candidate_packages(embedder, method_hint)
    module_candidates = _candidate_modules(embedder)

    package = None
    version = None

    for pkg in package_candidates:
        version = _metadata_version(pkg)
        if version:
            package = _normalize_package_name(pkg)
            break

    if version is None:
        for module in module_candidates:
            version = _module_version(module)
            if version:
                package = module
                break

    if version is None and "drnb" not in package_candidates:
        version = _metadata_version("drnb")
        if version:
            package = "drnb"

    if version is None:
        version = UNKNOWN_VALUE
    if package is None:
        package = UNKNOWN_VALUE

    return {"package": package, "version": version}


def get_embedder_version_info(
    embedder: Any, method_hint: str | None = None
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return version metadata for an embedder or list of embedders.

    The result is a dict with keys ``package``, ``version``, and ``source``
    (always ``"core"`` here). For chained embedders, a list of such dicts is
    returned, one per component.
    """
    if isinstance(embedder, list):
        name_parts: list[str] = []
        if method_hint and "+" in method_hint:
            name_parts = method_hint.split("+")
        results: list[dict[str, Any]] = []
        for idx, component in enumerate(embedder):
            part_hint = name_parts[idx] if idx < len(name_parts) else method_hint
            results.append(_resolve_single_version(component, part_hint))
        return results
    return _resolve_single_version(embedder, method_hint)
