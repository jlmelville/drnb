from dataclasses import dataclass

import openTSNE

import drnb.embed


@dataclass
class Tsne(drnb.embed.Embedder):
    def embed(self, x):
        return embed_tsne(x, self.embedder_kwds)


def embed_tsne(x, embedder_kwds):
    embedder = openTSNE.TSNE(n_components=2, **embedder_kwds)
    embedded = embedder.fit(x)

    return embedded
