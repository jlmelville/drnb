from dataclasses import dataclass

import numpy as np
import umap


@dataclass
class Umap:
    n_neighbors: int = 15
    init: str = "spectral"
    densmap: bool = False
    seed: int = None
    output_dens: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_umap(
            x,
            n_neighbors=self.n_neighbors,
            init=self.init,
            densmap=self.densmap,
            seed=self.seed,
            output_dens=self.output_dens,
        )


def embed_umap(
    x, n_neighbors=15, init="spectral", densmap=False, seed=None, output_dens=False
):
    umap_kwargs = {}
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        umap_kwargs["metric"] = "precomputed"

    embedder = umap.UMAP(
        random_state=seed,
        n_neighbors=n_neighbors,
        init=init,
        densmap=densmap,
        output_dens=output_dens,
        **umap_kwargs,
    )
    embedded = embedder.fit_transform(x)

    if densmap and output_dens:
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
