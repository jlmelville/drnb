#!/usr/bin/env python
from __future__ import annotations

import sys
from contextlib import redirect_stdout
from typing import Any, Literal

import numpy as np
import openTSNE
import openTSNE.nearest_neighbors as tsnenn
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.paths import (
    resolve_init_path,
    resolve_neighbors,
    resolve_x_path,
)
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin
from openTSNE import initialization as initialization_scheme

_PLUGIN_ONLY_PARAMS = {
    "use_precomputed_knn",
    "affinity",
    "symmetrize",
    "n_neighbors",
    "anneal_exaggeration",
    "n_exaggeration_iter",
    "n_anneal_steps",
    "anneal_momentum",
    "initial_momentum",
    "final_momentum",
    "gradient_descent_params",
}


def tsne_init(
    data: np.ndarray,
    affinities: openTSNE.affinity.Affinities,
    initialization: Literal["pca", "random", "spectral"] | np.ndarray | None = "pca",
    n_components: int = 2,
    random_state: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """t-SNE initialization using typical methods."""
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
    data: np.ndarray,
    affinities: openTSNE.affinity.Affinities,
    random_state: int = 42,
    n_exaggeration_iter: int = 125,
    early_exaggeration: float = 12.0,
    initial_momentum: float = 0.5,
    n_anneal_steps: int = 125,
    anneal_momentum: float = 0.8,
    n_iter: int = 500,
    final_momentum: float = 0.8,
    initialization: Literal["pca", "random", "spectral"] | np.ndarray | None = "pca",
    negative_gradient_method: Literal["auto", "bh", "fft"] = "auto",
    gradient_descent_params: dict = {},
) -> np.ndarray:
    """t-SNE with annealed exaggeration: after early exaggeration, slowly reduce
    the exaggeration factor to 1.0."""
    # initialization
    init = tsne_init(data, affinities, initialization, random_state=random_state)

    # Calculate the actual gradient method here to prevent spamming of
    # "Automatically determined negative gradient method" message inside the annealing
    # loop
    if negative_gradient_method == "auto":
        n_samples = affinities.P.shape[0]
        if n_samples < 10_000:
            negative_gradient_method = "bh"
        else:
            negative_gradient_method = "fft"

    E = openTSNE.TSNEEmbedding(
        init,
        affinities,
        random_state=random_state,
        negative_gradient_method=negative_gradient_method,
        **gradient_descent_params,
    )

    ## early exaggeration
    log(
        f"Running early exaggeration with exaggeration = {early_exaggeration:.2f} for {n_exaggeration_iter} iterations"
    )
    E = E.optimize(
        n_iter=n_exaggeration_iter,
        exaggeration=early_exaggeration,
        momentum=initial_momentum,
    )

    ## exaggeration annealing
    log(
        f"Annealing exaggeration from {early_exaggeration:.2f} to 1.0 over {n_anneal_steps} iterations"
    )
    exs = np.linspace(early_exaggeration, 1, n_anneal_steps)
    for ex in exs:
        E = E.optimize(
            n_iter=1,
            exaggeration=ex,
            momentum=anneal_momentum,
        )

    log(
        f"Running final optimization with momentum = {final_momentum:.2f} for {n_iter} iterations"
    )
    ## final optimization without exaggeration
    E = E.optimize(
        n_iter=n_iter,
        exaggeration=1,
        momentum=final_momentum,
    )

    return np.array(E)


def get_n_neighbors_for_perplexity(perplexity: float, x: np.ndarray) -> int:
    """Calculate the number of neighbors for a given perplexity."""
    n_samples = x.shape[0]
    k_neighbors = min(n_samples - 1, int(3 * perplexity))
    if k_neighbors < 3 * perplexity:
        log(
            f"Using k_neighbors {k_neighbors}, < 3 * perplexity {perplexity:.2f} may give unexpected results"
        )
    else:
        log(
            f"Using k_neighbors (no self) = {k_neighbors} with perplexity {perplexity:.2f}"
        )
    return k_neighbors


def get_tsne_affinities(
    affinity_type: Literal["perplexity", "uniform"],
    perplexity: float = 30.0,
    n_neighbors: int | None = None,
    x: np.ndarray | None = None,
    knn_index: tsnenn.PrecomputedNeighbors | None = None,
    metric: str = "euclidean",
    symmetrize: Literal["max", "mean", "none"] = "mean",
) -> openTSNE.affinity.Affinities:
    """Get t-SNE affinities, using either perplexity-based or uniform affinities."""

    if affinity_type == "perplexity":
        if n_neighbors is None:
            n_neighbors = get_n_neighbors_for_perplexity(perplexity, x)
    elif affinity_type == "uniform":
        if n_neighbors is None:
            raise ValueError("n_neighbors cannot be None")
        log(f"Calculating uniform affinities with n_neighbors = {n_neighbors}")
    else:
        raise ValueError(f"Unknown affinity type '{affinity_type}'")

    if affinity_type == "perplexity":
        return openTSNE.affinity.PerplexityBasedNN(
            perplexity=perplexity,
            knn_index=knn_index,
            symmetrize=symmetrize,
            data=x if knn_index is None else None,
            metric=metric,
        )
    return openTSNE.affinity.Uniform(
        knn_index=knn_index,
        symmetrize=symmetrize,
        data=x if knn_index is None else None,
        metric=metric,
        k_neighbors=n_neighbors,
    )


def _load_initialization(req: sdk_protocol.PluginRequest) -> np.ndarray | str | None:
    init_path = resolve_init_path(req)
    if init_path:
        return np.load(init_path, allow_pickle=False)
    params = req.params or {}
    return params.get("initialization")


def _build_affinities(req: sdk_protocol.PluginRequest, x: np.ndarray) -> Any:
    if not req.options.use_precomputed_knn:
        return None
    params = req.params or {}
    neighbors = resolve_neighbors(req)
    knn_index: tsnenn.PrecomputedNeighbors | None = None
    if neighbors and neighbors.idx_path:
        try:
            idx = np.load(neighbors.idx_path, allow_pickle=False)
            if idx.shape[1] > 1:
                idx = idx[:, 1:]  # drop self neighbor
            dist = (
                np.load(neighbors.dist_path, allow_pickle=False)
                if neighbors.dist_path
                else None
            )
            if dist is not None and dist.shape[1] > 1:
                dist = dist[:, 1:]
            knn_index = tsnenn.PrecomputedNeighbors(idx, dist)
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to load precomputed neighbors: {exc}")
            knn_index = None

    return get_tsne_affinities(
        affinity_type=params.get("affinity", "perplexity"),
        perplexity=params.get("perplexity", 30.0),
        n_neighbors=params.get("n_neighbors"),
        x=x,
        metric=params.get("metric", "euclidean"),
        symmetrize=params.get("symmetrize", "mean"),
        knn_index=knn_index,
    )


def run_tsne(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    init = _load_initialization(req)
    with redirect_stdout(sys.stderr):
        affinities = _build_affinities(req, x)

    anneal = bool(params.get("anneal_exaggeration", False))

    with redirect_stdout(sys.stderr):
        if anneal:
            if affinities is None:
                raise RuntimeError(
                    "Annealed exaggeration requires precomputed affinities"
                )
            early_exaggeration_iter = params.get("early_exaggeration_iter", 250)
            n_exaggeration_iter = int(early_exaggeration_iter / 2)
            n_anneal_steps = n_exaggeration_iter
            anneal_momentum = params.get(
                "anneal_momentum", params.get("final_momentum", 0.8)
            )
            gradient_descent_params = {
                "n_jobs": params.get("n_jobs", 1),
                "dof": params.get("dof", 1),
                "learning_rate": params.get("learning_rate", "auto"),
                "verbose": params.get("verbose", False),
                "theta": params.get("theta", 0.5),
                "n_interpolation_points": params.get("n_interpolation_points", 3),
                "min_num_intervals": params.get("min_num_intervals", 50),
                "ints_in_interval": params.get("ints_in_interval", 1),
                "max_grad_norm": params.get("max_grad_norm", None),
                "max_step_norm": params.get("max_step_norm", 5),
                "callbacks": params.get("callbacks", None),
                "callbacks_every_iters": params.get("callbacks_every_iters", 50),
            }
            embedded = tsne_annealed_exaggeration(
                data=x,
                affinities=affinities,
                random_state=params.get("random_state", 42),
                n_exaggeration_iter=n_exaggeration_iter,
                early_exaggeration=params.get("early_exaggeration", 12.0),
                initial_momentum=params.get("initial_momentum", 0.5),
                n_anneal_steps=n_anneal_steps,
                anneal_momentum=anneal_momentum,
                n_iter=params.get("n_iter", 500),
                final_momentum=params.get("final_momentum", 0.8),
                initialization=init,
                negative_gradient_method=params.get("negative_gradient_method", "auto"),
                gradient_descent_params=gradient_descent_params,
            )
        else:
            tsne_params = {
                key: value
                for key, value in params.items()
                if key not in _PLUGIN_ONLY_PARAMS
            }
            log(f"Running openTSNE.TSNE with params={summarize_params(tsne_params)}")
            tsne = openTSNE.TSNE(n_components=2, **tsne_params)
            embedded = tsne.fit(x, affinities=affinities, initialization=init)

    coords = np.asarray(embedded, dtype=np.float32, order="C")
    return save_result_npz(req.output.result_path, coords)


if __name__ == "__main__":
    run_plugin({"tsne": run_tsne})
