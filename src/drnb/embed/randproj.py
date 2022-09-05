# Random Projections
from dataclasses import dataclass

import sklearn.random_projection

import drnb.embed
from drnb.log import log


@dataclass
class RandProj(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_randproj(x, params)


def embed_randproj(
    x,
    params,
):
    log.info("Running Sparse Random Projection")
    embedder = sklearn.random_projection.SparseRandomProjection(
        n_components=2, **params
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
