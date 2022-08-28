from dataclasses import dataclass

import pacmap


@dataclass
class Pacmap:
    n_neighbors: int = None
    init: str = "pca"
    seed: int = None
    intermediate: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_pacmap(
            x,
            n_neighbors=self.n_neighbors,
            init=self.init,
            seed=self.seed,
            intermediate=self.intermediate,
        )


def embed_pacmap(x, n_neighbors=None, init="pca", seed=None, intermediate=False):
    embedder = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=seed,
        intermediate=intermediate,
    )
    result = embedder.fit_transform(x, init=init)
    if intermediate:
        embedded = dict(coords=result[-1])
        for i in range(result.shape[0]):
            embedded[f"it_{embedder.intermediate_snapshots[i]}"] = result[i]
    else:
        embedded = result

    return embedded
