from dataclasses import dataclass

import openTSNE
import openTSNE.nearest_neighbors as tsnenn

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log


def get_n_neighbors_for_perplexity(perplexity, x):
    n_samples = x.shape[0]
    k_neighbors = min(n_samples - 1, int(3 * perplexity))
    if k_neighbors < 3 * perplexity:
        log.info(
            "Using k_neighbors %d, < 3 * perplexity %.2f "
            "may give unexpected results",
            k_neighbors,
            perplexity,
        )
    else:
        log.info(
            "Using k_neighbors (no self) = %d with perplexity %.2f",
            k_neighbors,
            perplexity,
        )
    return k_neighbors


def get_tsne_affinities(
    affinity_type,
    perplexity=30,
    n_neighbors=None,
    x=None,
    knn_params=None,
    metric="euclidean",
    ctx=None,
):
    if knn_params is None:
        knn_params = {}
    if affinity_type == "perplexity":
        if n_neighbors is None:
            n_neighbors = get_n_neighbors_for_perplexity(perplexity, x)
    elif affinity_type == "uniform":
        if n_neighbors is None:
            raise ValueError("n_neighbors cannot be None")
        log.info(
            "Calculating uniform affinities with n_neighbors = %d",
            n_neighbors,
        )
    else:
        raise ValueError(f"Unknown affinity type '{affinity_type}'")
    # openTSNE does not use self index so ask for one more
    precomputed_knn = knn.get_neighbors_with_ctx(
        x, metric, n_neighbors + 1, knn_params=knn_params, ctx=ctx
    )
    tsne_knn = tsnenn.PrecomputedNeighbors(
        precomputed_knn.idx[:, 1:], precomputed_knn.dist[:, 1:]
    )

    if affinity_type == "perplexity":
        return openTSNE.affinity.PerplexityBasedNN(
            perplexity=perplexity,
            knn_index=tsne_knn,
        )
    return openTSNE.affinity.Uniform(knn_index=tsne_knn)


@dataclass
class Tsne(drnb.embed.Embedder):
    use_precomputed_knn: bool = True
    initialization: str = None
    n_neighbors: int = None
    affinity: str = "perplexity"

    def embed_impl(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            affinities = get_tsne_affinities(
                affinity_type=self.affinity,
                perplexity=params.get("perplexity", 30),
                n_neighbors=self.n_neighbors,
                x=x,
                knn_params=knn_params,
                metric=params.get("metric", "euclidean"),
                ctx=ctx,
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
