from dataclasses import dataclass

import numpy as np
import umap

import drnb.embed


@dataclass
class Umap(drnb.embed.Embedder):
    def embed(self, x):
        return embed_umap(x, self.embedder_kwds)


def embed_umap(
    x,
    embedder_kwds,
):
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        embedder_kwds["metric"] = "precomputed"

    embedder = umap.UMAP(
        **embedder_kwds,
    )
    embedded = embedder.fit_transform(x)

    if embedder_kwds.get("densmap", False) and embedder_kwds.get("output_dens", False):
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
