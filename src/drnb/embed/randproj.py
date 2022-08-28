# Random Projections
from dataclasses import dataclass

import sklearn.random_projection


@dataclass
class RandProj:
    seed: int = True

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return randproj(x, seed=self.seed)


def randproj(
    x,
    seed=None,
):
    embedder = sklearn.random_projection.SparseRandomProjection(
        random_state=seed, n_components=2
    )
    embedded = embedder.fit_transform(x)
    return embedded
