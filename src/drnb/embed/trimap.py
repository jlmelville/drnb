from dataclasses import dataclass

import trimap


@dataclass
class Trimap:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return trimap_embed(x)


def trimap_embed(x):
    embedder = trimap.TRIMAP(
        n_dims=2,
    )
    embedded = embedder.fit_transform(x)

    return embedded
