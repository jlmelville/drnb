from dataclasses import dataclass

import sklearn.manifold

import drnb.embed
from drnb.log import log


@dataclass
class Mmds(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_mmds(x, params)


def embed_mmds(
    x,
    params,
):
    log.info("Running sklearn MDS")
    embedder = sklearn.manifold.MDS(
        n_components=2, metric=True, normalized_stress=False, **params
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
