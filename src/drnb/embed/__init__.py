def create_embedder(method, embed_kwargs=None):
    if method == "randproj":
        # pylint: disable=import-outside-toplevel
        import drnb.embed.randproj

        ctor = drnb.embed.randproj.RandProj
    elif method == "tsne":
        # pylint: disable=import-outside-toplevel
        import drnb.embed.tsne

        ctor = drnb.embed.tsne.Tsne
    else:
        raise ValueError(f"Unknown method {method}")

    if embed_kwargs is None:
        embed_kwargs = {}

    embedder = ctor(**embed_kwargs)
    return embedder
