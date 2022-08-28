# pylint: disable=import-outside-toplevel
def create_embedder(method, embed_kwargs=None):
    method = method.lower()
    if method == "ncvis":
        import drnb.embed.ncvis

        ctor = drnb.embed.ncvis.NCVis
    elif method == "randproj":
        import drnb.embed.randproj

        ctor = drnb.embed.randproj.RandProj
    elif method == "tsne":
        import drnb.embed.tsne

        ctor = drnb.embed.tsne.Tsne
    elif method == "trimap":
        import drnb.embed.trimap

        ctor = drnb.embed.trimap.Trimap
    else:
        raise ValueError(f"Unknown method {method}")

    if embed_kwargs is None:
        embed_kwargs = {}

    embedder = ctor(**embed_kwargs)
    return embedder
