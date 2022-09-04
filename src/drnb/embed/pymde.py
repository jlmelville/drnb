from dataclasses import dataclass

import pymde
import torch

import drnb.embed


@dataclass
class Pymde(drnb.embed.Embedder):
    seed: int = None

    def embed_impl(self, x, params, ctx=None):
        return embed_pymde_nbrs(x, self.seed, params)


def embed_pymde_nbrs(
    x,
    seed,
    params,
):
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)

    embedder = pymde.preserve_neighbors(x, device=device, embedding_dim=2, **params)
    embedded = embedder.embed().cpu().data.numpy()

    return embedded
