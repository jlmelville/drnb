from dataclasses import dataclass

import sklearn.decomposition

import drnb.embed
from drnb.log import log
from drnb.preprocess import center


@dataclass
class Tsvd(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_tsvd(x, params)


def embed_tsvd(x, params):
    x = center(x)

    log.info("Running Truncated SVD")
    embedder = sklearn.decomposition.TruncatedSVD(
        n_components=2,
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")
    
    return embedded
