from dataclasses import dataclass

import openTSNE


@dataclass
class Tsne:
    seed: int = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return tsne_embed(x, seed=self.seed)


def tsne_embed(
    x,
    seed=None,
):
    embedder = openTSNE.TSNE(n_components=2, random_state=seed)
    embedded = embedder.fit(x)

    return embedded
