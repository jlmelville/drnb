from dataclasses import dataclass

import openTSNE

import drnb.embed


@dataclass
class Tsne(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_tsne(x, params)


def embed_tsne(x, params):
    embedder = openTSNE.TSNE(n_components=2, **params)
    embedded = embedder.fit(x)

    return embedded
