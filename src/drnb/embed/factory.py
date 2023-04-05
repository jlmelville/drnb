# pylint: disable=import-outside-toplevel,too-many-statements
def create_embedder(method, embed_kwds=None):
    if isinstance(method, list):
        return [create_embedder(m) for m in method]

    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        embed_kwds = method[1]
        method = method[0]

    if embed_kwds is None:
        embed_kwds = dict(params={})

    if "params" not in embed_kwds or embed_kwds["params"] is None:
        embed_kwds["params"] = {}

    # already created embedder factory
    if callable(method):
        ctor = method
    else:
        method = method.lower()

        if method == "ncvis":
            from drnb.embed.ncvis import NCVis as ctor
        elif method == "pacmap":
            from drnb.embed.pacmap import Pacmap as ctor
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
            from drnb.embed.spacemap import Spacemap as ctor
        elif method == "skmmds":
            from drnb.embed.mmds import Mmds as ctor
        elif method == "mmds":
            from drnb.embed.smmds import Mmds as ctor
        elif method == "rescale":
            from drnb.embed.rescale import Rescale as ctor
        elif method == "umato":
            from drnb.embed.umato import Umato as ctor
        else:
            raise ValueError(f"Unknown method {method}")

    embedder = ctor(**embed_kwds)
    return embedder
