from dataclasses import dataclass

import sklearn.decomposition

import drnb.embed


@dataclass
class Pca(drnb.embed.Embedder):
    def embed(self, x, ctx=None):
        return embed_pca(x, self.embedder_kwds)


def embed_pca(
    x,
    embedder_kwds,
):
    embedder = sklearn.decomposition.PCA(n_components=2, **embedder_kwds)
    embedded = embedder.fit_transform(x)

    return embedded
