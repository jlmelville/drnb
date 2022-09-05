from dataclasses import dataclass

import openTSNE

import drnb.embed

from drnb.log import log


@dataclass
class Tsne(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_tsne(x, params)


def embed_tsne(x, params):
    log.info("Running t-SNE")
    embedder = openTSNE.TSNE(n_components=2, **params)
    embedded = embedder.fit(x)
    log.info("Embedding completed")

    return embedded
