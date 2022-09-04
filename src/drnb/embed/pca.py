from dataclasses import dataclass

import sklearn.decomposition

import drnb.embed


@dataclass
class Pca(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_pca(x, params)


def embed_pca(
    x,
    params,
):
    embedder = sklearn.decomposition.PCA(n_components=2, **params)
    embedded = embedder.fit_transform(x)

    return embedded
