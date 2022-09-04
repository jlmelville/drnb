# Random Projections
from dataclasses import dataclass

import sklearn.random_projection

import drnb.embed


@dataclass
class RandProj(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_randproj(x, params)


def embed_randproj(
    x,
    params,
):
    embedder = sklearn.random_projection.SparseRandomProjection(
        n_components=2, **params
    )
    embedded = embedder.fit_transform(x)
    return embedded
