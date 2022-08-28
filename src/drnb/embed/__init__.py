# pylint: disable=import-outside-toplevel
def create_embedder(method, embed_kwargs=None):
    if embed_kwargs is None:
        embed_kwargs = {}

    method = method.lower()

    if method == "densmap":
        from drnb.embed.umap import Umap as ctor

        embed_kwargs["densmap"] = True
        if "n_neighbors" not in embed_kwargs:
            embed_kwargs["n_neighbors"] = 30
    elif method == "ncvis":
        from drnb.embed.ncvis import NCVis as ctor
    elif method == "pca":
        from drnb.embed.pca import Pca as ctor
    elif method == "pymde":
        from drnb.embed.pymde import Pymde as ctor
    elif method == "randproj":
        from drnb.embed.randproj import RandProj as ctor
    elif method == "tsne":
        from drnb.embed.tsne import Tsne as ctor
    elif method == "tsvd":
        from drnb.embed.tsvd import Tsvd as ctor
    elif method == "trimap":
        from drnb.embed.trimap import Trimap as ctor
    elif method == "umap":
        from drnb.embed.umap import Umap as ctor
    else:
        raise ValueError(f"Unknown method {method}")

    embedder = ctor(**embed_kwargs)
    return embedder
