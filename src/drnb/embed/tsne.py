from dataclasses import dataclass

import numpy as np
import openTSNE
import openTSNE.nearest_neighbors as tsnenn
from openTSNE import initialization as initialization_scheme

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log


def tsne_init(
    data,
    affinities,
    initialization="pca",
    n_components=2,
    random_state=42,
    verbose=False,
):
    if initialization is None:
        initialization = "pca"

    n_samples = data.shape[0]

    if isinstance(initialization, np.ndarray):
        embedding = np.array(initialization)

        stddev = np.std(embedding, axis=0)
        if any(stddev > 1e-2):
            log.warning(
                "Standard deviation of embedding is greater than 0.0001. Initial "
                "embeddings with high variance may have display poor convergence."
            )

    elif initialization == "pca":
        embedding = initialization_scheme.pca(
            data,
            n_components,
            random_state=random_state,
            verbose=verbose,
        )
    elif initialization == "random":
        embedding = initialization_scheme.random(
            n_samples,
            n_components,
            random_state=random_state,
            verbose=verbose,
        )
    elif initialization == "spectral":
        embedding = initialization_scheme.spectral(
            affinities.P,
            n_components,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown tsne initialization: {initialization}")
    return embedding


# https://github.com/berenslab/pubmed-landscape/blob/eb963c42627da7439dffe1e962a404f76bc905ad/scripts/BERT-based-embeddings/05-rgm-pipeline-TFIDF-1M.ipynb#L31
def tsne_annealed_exaggeration(
    data,
    affinities,
    random_state=42,
    n_exaggeration_iter=125,
    early_exaggeration=12,
    initial_momentum=0.5,
    n_anneal_steps=125,
    anneal_momentum=0.8,
    n_iter=500,
    final_momentum=0.8,
    initialization="pca",
):
    # initialization
    init = tsne_init(data, affinities, initialization, random_state=random_state)

    # prevent spamming of "Automatically determined negative gradient method" message
    n_samples = affinities.P.shape[0]
    if n_samples < 10_000:
        negative_gradient_method = "bh"
    else:
        negative_gradient_method = "fft"

    E = openTSNE.TSNEEmbedding(
        init,
        affinities,
        n_jobs=-1,
        random_state=random_state,
        negative_gradient_method=negative_gradient_method,
    )

    ## early exaggeration
    E = E.optimize(
        n_iter=n_exaggeration_iter,
        exaggeration=early_exaggeration,
        momentum=initial_momentum,
        n_jobs=-1,
    )

    ## exaggeration annealing
    exs = np.linspace(early_exaggeration, 1, n_anneal_steps)
    for ex in exs:
        E = E.optimize(
            n_iter=1,
            exaggeration=ex,
            momentum=anneal_momentum,
            n_jobs=-1,
        )

    ## final optimization without exaggeration
    E = E.optimize(n_iter=n_iter, exaggeration=1, momentum=final_momentum, n_jobs=-1)

    return np.array(E)


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
    precomputed_init: np.ndarray = None
    anneal_exaggeration: bool = False

    def embed_impl(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        affinities = None
        if self.use_precomputed_knn and ctx is not None:
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

        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            init = self.precomputed_init
        elif self.initialization is not None:
            log.info("Using '%s' initialization", self.initialization)
            init = self.initialization
        else:
            init = None

        return embed_tsne(
            x,
            params,
            affinities=affinities,
            initialization=init,
            anneal_exagg=self.anneal_exaggeration,
        )


def embed_tsne(x, params, affinities=None, initialization=None, anneal_exagg=False):
    log.info("Running t-SNE")

    if anneal_exagg:
        if affinities is None:
            raise ValueError(
                "Annealed exaggeration is only supported with pre-calculated affinities"
            )
        random_state = params.get("random_state", 42)
        early_exaggeration_iter = params.get("early_exaggeration_iter", 250)
        n_exaggeration_iter = int(early_exaggeration_iter / 2)
        early_exaggeration = params.get("early_exaggeration", 12)
        initial_momentum = params.get("initial_momentum", 0.5)
        final_momentum = params.get("final_momentum", 0.8)
        n_iter = params.get("n_iter", 500)
        embedded = tsne_annealed_exaggeration(
            data=x,
            affinities=affinities,
            random_state=random_state,
            n_exaggeration_iter=n_exaggeration_iter,
            early_exaggeration=early_exaggeration,
            initial_momentum=initial_momentum,
            n_anneal_steps=n_exaggeration_iter,
            anneal_momentum=final_momentum,
            n_iter=n_iter,
            final_momentum=final_momentum,
            initialization=initialization,
        )
    else:
        embedder = openTSNE.TSNE(n_components=2, **params)
        embedded = embedder.fit(x, affinities=affinities, initialization=initialization)
    log.info("Embedding completed")

    return embedded
