from dataclasses import dataclass

import pymde
import torch


@dataclass
class Pymde:
    n_neighbors: int = None
    seed: int = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return pymde_nbrs(x, n_neighbors=self.n_neighbors, seed=self.seed)


def pymde_nbrs(
    x,
    n_neighbors=None,
    seed=None,
):
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)
    embedder = pymde.preserve_neighbors(
        x, device=device, embedding_dim=2, n_neighbors=n_neighbors
    )
    embedded = embedder.embed().cpu().data.numpy()

    return embedded
