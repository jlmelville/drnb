from dataclasses import fields as dataclass_fields
from typing import Callable

import drnb.embed.base
from drnb.plugins.external import ExternalEmbedder
from drnb.plugins.registry import get_registry, plugins_enabled
from drnb.types import ActionConfig


def create_embedder(
    method: ActionConfig | list | Callable, embed_kwds: dict | None = None
) -> drnb.embed.base.Embedder:
    """Create an embedder from an action configuration, a list of action configurations,
    or a callable embedder factory/constructor."""
    if isinstance(method, list):
        return [create_embedder(m) for m in method]

    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        embed_kwds = method[1]
        method = method[0]

    if embed_kwds is None:
        embed_kwds = {"params": {}}

    if "params" not in embed_kwds or embed_kwds["params"] is None:
        embed_kwds["params"] = {}

    # already created embedder factory
    if callable(method):
        ctor = method
    else:
        ctor = _str_to_ctor(method)

    embedder = ctor(**embed_kwds)
    return embedder


# pylint: disable=import-outside-toplevel,too-many-statements
def _str_to_ctor(method: str) -> drnb.embed.base.Embedder:
    method = method.lower()

    if plugins_enabled():
        entry = get_registry().lookup(method)
        if entry is not None:

            allowed = {f.name for f in dataclass_fields(ExternalEmbedder)}

            def _ctor(**embed_kwds):
                normalized = _normalize_external_kwds(embed_kwds, allowed)
                return ExternalEmbedder(method=method, **normalized)

            return _ctor

    if method == "ncvis":
        from drnb.embed.ncvis import NCVis as ctor
    # elif method == "pacmap":
    #     from drnb.embed.pacmap import Pacmap as ctor
    elif method == "pca":
        from drnb.embed.pca import Pca as ctor
    elif method == "pymde":
        from drnb.embed.pymde import Pymde as ctor
    elif method == "randproj":
        from drnb.embed.randproj import RandProj as ctor
    elif method == "trimap":
        from drnb.embed.trimap import Trimap as ctor
    elif method == "tsne":
        from drnb.embed.tsne import Tsne as ctor
    elif method == "tsvd":
        from drnb.embed.tsvd import Tsvd as ctor
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
    elif method == "localmap":
        from drnb.embed.pacmap import Localmap as ctor
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
        from drnb.embed.spacemap import Spacemap as ctor
    elif method == "mmds":
        from drnb.embed.smmds import Mmds as ctor
    elif method == "rescale":
        from drnb.embed.rescale import Rescale as ctor
    elif method == "tsne-rescale":
        from drnb.embed.rescale import TsneRescale as ctor
    elif method == "umato":
        from drnb.embed.umato import Umato as ctor
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


def _normalize_external_kwds(
    embed_kwds: dict | None, allowed: set[str]
) -> dict:
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
