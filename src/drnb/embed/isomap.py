from dataclasses import dataclass

import sklearn.manifold

import drnb.embed
from drnb.log import log


@dataclass
class Isomap(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_isomap(x, params)


def embed_isomap(
    x,
    params,
):
    log.info("Running Isomap")
    embedder = sklearn.manifold.Isomap(n_components=2, **params)
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
