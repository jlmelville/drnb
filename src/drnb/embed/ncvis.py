from dataclasses import dataclass

import ncvis


@dataclass
class NCVis:
    seed: int = None
    n_neighbors: int = 30

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return ncvis_embed(x, seed=self.seed, n_neighbors=self.n_neighbors)


def ncvis_embed(x, seed=None, n_neighbors=30):

    kwargs = dict(n_neighbors=n_neighbors)
    if isinstance(seed, int):
        kwargs["random_seed"] = seed
    embedder = ncvis.NCVis(**kwargs)
    embedded = embedder.fit_transform(x)

    return embedded
