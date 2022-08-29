from dataclasses import dataclass

import pymde
import torch

import drnb.embed


@dataclass
class Pymde(drnb.embed.Embedder):
    def embed(self, x):
        return embed_pymde_nbrs(x, self.embedder_kwds)


def embed_pymde_nbrs(
    x,
    embedder_kwds,
):
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if embedder_kwds.get("seed") is not None:
        pymde.seed(embedder_kwds["seed"])
        del embedder_kwds["seed"]

    embedder = pymde.preserve_neighbors(
        x, device=device, embedding_dim=2, **embedder_kwds
    )
    embedded = embedder.embed().cpu().data.numpy()

    return embedded
