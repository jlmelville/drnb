from dataclasses import fields as dataclass_fields
from typing import Callable

import drnb.embed.base
from drnb.plugins.external import ExternalEmbedder
from drnb.plugins.registry import get_registry
from drnb.types import ActionConfig, EmbedConfig


def create_single_embedder(
    embed_config: EmbedConfig | ActionConfig | Callable, embed_kwds: dict | None = None
) -> drnb.embed.base.Embedder:
    """Create a single embedder from an EmbedConfig, action configuration, or callable.

    This is the core function that creates a single embedder instance.
    For lists, use create_embedder() which handles dispatching.
    """
    # Handle EmbedConfig
    if isinstance(embed_config, EmbedConfig):
        # Merge embed_kwds with config's wrapper_kwds and params
        if embed_kwds is None:
            embed_kwds = {}
        # Merge wrapper_kwds (embed_kwds overrides)
        merged_wrapper_kwds = {**embed_config.wrapper_kwds, **embed_kwds}
        # Merge params (embed_kwds["params"] overrides if present)
        merged_params = embed_config.params.copy()
        if "params" in embed_kwds:
            merged_params.update(embed_kwds["params"])
        final_kwds = {**merged_wrapper_kwds, "params": merged_params}
        method_name = embed_config.name
    # Handle tuple (backward compatibility)
    elif isinstance(embed_config, tuple):
        if len(embed_config) != 2:
            raise ValueError("Unexpected format for method")
        final_kwds = embed_config[1].copy()
        if embed_kwds is not None:
            # Merge embed_kwds
            if "params" in embed_kwds:
                final_kwds.setdefault("params", {}).update(embed_kwds["params"])
            final_kwds.update({k: v for k, v in embed_kwds.items() if k != "params"})
        method_name = embed_config[0]
    # Handle callable
    elif callable(embed_config):
        if embed_kwds is None:
            embed_kwds = {"params": {}}
        if "params" not in embed_kwds or embed_kwds["params"] is None:
            embed_kwds["params"] = {}
        return embed_config(**embed_kwds)
    # Handle string
    else:
        if embed_kwds is None:
            embed_kwds = {"params": {}}
        if "params" not in embed_kwds or embed_kwds["params"] is None:
            embed_kwds["params"] = {}
        final_kwds = embed_kwds
        method_name = embed_config

    # Get constructor and create embedder
    ctor = _str_to_ctor(method_name)
    return ctor(**final_kwds)


def create_embedder(
    method: EmbedConfig | ActionConfig | list | Callable, embed_kwds: dict | None = None
) -> drnb.embed.base.Embedder | list[drnb.embed.base.Embedder]:
    """Create an embedder from an EmbedConfig, action configuration, a list of configurations,
    or a callable embedder factory/constructor.

    If method is a list, returns a list of embedders.
    Otherwise, returns a single embedder.
    """
    if isinstance(method, list):
        return [create_single_embedder(m) for m in method]

    return create_single_embedder(method, embed_kwds)


# pylint: disable=import-outside-toplevel,too-many-statements
def _str_to_ctor(method: str) -> drnb.embed.base.Embedder:
    method = method.lower()

    entry = get_registry().lookup(method)
    if entry is not None:
        allowed = {f.name for f in dataclass_fields(ExternalEmbedder)}

        def _ctor(**embed_kwds):
            normalized = _normalize_external_kwds(embed_kwds, allowed)
            return ExternalEmbedder(method=method, **normalized)

        return _ctor

    if method == "pca":
        from drnb.embed.pca import Pca as ctor
    elif method == "randproj":
        from drnb.embed.randproj import RandProj as ctor
    elif method == "umap":
        from drnb.embed.umap import Umap as ctor
    elif method == "negumap":
        from drnb.embed.umap.negumap import NegUmap as ctor
    elif method == "negtumap":
        from drnb.embed.umap.negtumap import NegTumap as ctor
    elif method == "negtsne":
        from drnb.embed.umap.negtsne import NegTsne as ctor
    elif method == "umapspectral":
        from drnb.embed.umap.spectral import UmapSpectral as ctor
    elif method == "bgspectral":
        from drnb.embed.umap.spectral import BinaryGraphSpectral as ctor
    elif method == "umap2":
        from drnb.embed.umap.custom2 import Umap2 as ctor
    elif method == "ivhd":
        from drnb.embed.ivhd import Ivhd as ctor
    elif method == "xvhd":
        from drnb.embed.ivhd import Xvhd as ctor
    elif method == "pacumap":
        from drnb.embed.umap.pacumap import Pacumap as ctor
    elif method == "htumap":
        from drnb.embed.umap.htumap import Htumap as ctor
    elif method == "htnegumap":
        from drnb.embed.umap.htumap import Htnegumap as ctor
    elif method == "smmds":
        from drnb.embed.smmds import Smmds as ctor
    elif method == "snmds":
        from drnb.embed.smmds import Snmds as ctor
    elif method == "leopold":
        from drnb.embed.leopold import Leopold as ctor
    elif method == "spacemap":
        from drnb.embed.deprecated.spacemap import Spacemap as ctor
    elif method == "mmds":
        from drnb.embed.smmds import Mmds as ctor
    elif method == "rescale":
        from drnb.embed.rescale import Rescale as ctor
    elif method == "tsne-rescale":
        from drnb.embed.rescale import TsneRescale as ctor
    elif method == "sklearn-mmds":
        from drnb.embed.mmds import Mmds as ctor
    elif method == "sklearn-nmds":
        from drnb.embed.mmds import Nmds as ctor
    elif method == "isomap":
        from drnb.embed.isomap import Isomap as ctor
    elif method == "skmmds":
        from drnb.embed.skmmds import Skmmds as ctor
    elif method == "sikmmds":
        from drnb.embed.skmmds import Sikmmds as ctor
    elif method == "rsikmmds":
        from drnb.embed.skmmds import Rsikmmds as ctor
    elif method == "mrsikmmds":
        from drnb.embed.skmmds import Mrsikmmds as ctor
    elif method == "lcmmds":
        from drnb.embed.skmmds import Lcmmds as ctor
    else:
        raise ValueError(f"Unknown method {method}")
    return ctor


def _normalize_external_kwds(embed_kwds: dict | None, allowed: set[str]) -> dict:
    if embed_kwds is None:
        embed_kwds = {}
    embed_kwds = dict(embed_kwds)
    params = embed_kwds.setdefault("params", {})
    if params is None:
        params = {}
        embed_kwds["params"] = params
    if not isinstance(params, dict):
        raise ValueError("External embedder params must be a dict")
    for key in list(embed_kwds.keys()):
        if key == "params":
            continue
        if key not in allowed:
            params[key] = embed_kwds.pop(key)
    return embed_kwds
