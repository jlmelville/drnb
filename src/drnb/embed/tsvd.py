from dataclasses import dataclass

import sklearn.decomposition

import drnb.embed
from drnb.preprocess import center


@dataclass
class Tsvd(drnb.embed.Embedder):
    def embed(self, x, ctx=None):
        return embed_tsvd(x, self.embedder_kwds)


def embed_tsvd(x, embedder_kwds):
    x = center(x)

    embedder = sklearn.decomposition.TruncatedSVD(
        n_components=2,
        **embedder_kwds,
    )
    embedded = embedder.fit_transform(x)

    return embedded
