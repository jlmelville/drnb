import ncvis

import drnb.embed


class NCVis(drnb.embed.Embedder):
    def embed(self, x):
        return ncvis_embed(x, self.embedder_kwds)


# * d=2
# * n_threads=-1
# * n_neighbors=15
# * M=16
# * ef_construction=200
# * random_seed=42
# * n_epochs=50
# * n_init_epochs=20
# * spread=1.0
# * min_dist=0.4
# * a=None
# * b=None
# * alpha=1.0,
# * alpha_Q=1.0,
# * n_noise=None
# * distance="euclidean"


def ncvis_embed(x, embedder_kwds):
    embedder = ncvis.NCVis(**embedder_kwds)
    embedded = embedder.fit_transform(x)

    return embedded
