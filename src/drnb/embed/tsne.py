from dataclasses import dataclass

import openTSNE

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log


# Dummy knn class that cannot be used for transforming new data (but that's ok)
class PrecomputedKNNIndex:
    def __init__(self, indices, distances):
        self.indices = indices
        self.distances = distances
        self.k = indices.shape[1]

    def build(self):
        return self.indices, self.distances

    def query(self, query, k):
        raise NotImplementedError("No query with a pre-computed knn")

    def check_metric(self, metric):
        if callable(metric):
            pass

        return metric


@dataclass
class Tsne(drnb.embed.Embedder):
    use_precomputed_knn: bool = True
    initialization: str = None

    def embed_impl(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_samples = x.shape[0]
            perplexity = params.get("perplexity", 30)
            n_neighbors = min(n_samples - 1, int(3 * perplexity))
            log.info(
                "Using n_neighbors (no self) = %d based on perplexity %.2f",
                n_neighbors,
                perplexity,
            )
            # openTSNE does not use self index so ask for one more
            precomputed_knn = knn.get_neighbors_with_ctx(
                x, metric, n_neighbors + 1, knn_params=knn_params, ctx=ctx
            )
            tsne_knn = PrecomputedKNNIndex(
                precomputed_knn.idx[:, 1:], precomputed_knn.dist[:, 1:]
            )

            log.info("Calculating affinity for perplexity %.2f", perplexity)
            affinities = openTSNE.affinity.PerplexityBasedNN(
                perplexity=perplexity,
                knn_index=tsne_knn,
            )
        if self.initialization is not None:
            log.info("Using '%s' initialization", self.initialization)

        return embed_tsne(
            x, params, affinities=affinities, initialization=self.initialization
        )


def embed_tsne(x, params, affinities=None, initialization=None):
    log.info("Running t-SNE")
    embedder = openTSNE.TSNE(n_components=2, **params)
    embedded = embedder.fit(x, affinities=affinities, initialization=initialization)
    log.info("Embedding completed")

    return embedded
