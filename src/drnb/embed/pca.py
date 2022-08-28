from dataclasses import dataclass

import sklearn.decomposition


@dataclass
class Pca:
    seed: int = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_pca(x, seed=self.seed)


def embed_pca(
    x,
    seed=None,
):
    embedder = sklearn.decomposition.PCA(
        random_state=seed,
        n_components=2,
    )
    embedded = embedder.fit_transform(x)

    return embedded
