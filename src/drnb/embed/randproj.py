# Random Projections
from dataclasses import dataclass

import sklearn.random_projection

import drnb.embed


@dataclass
class RandProj(drnb.embed.Embedder):
    def embed(self, x):
        return embed_randproj(x, self.embedder_kwds)


def embed_randproj(
    x,
    embedder_kwds,
):
    embedder = sklearn.random_projection.SparseRandomProjection(
        n_components=2, **embedder_kwds
    )
    embedded = embedder.fit_transform(x)
    return embedded
