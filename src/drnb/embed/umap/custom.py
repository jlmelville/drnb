from typing import Literal

import numba
import numpy as np
import scipy.sparse
import umap
import umap.distances

# pylint: disable=no-name-in-module
from numpy.random import RandomState
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from umap.layouts import _optimize_layout_euclidean_single_epoch
from umap.spectral import spectral_layout
from umap.umap_ import INT32_MAX, INT32_MIN, make_epochs_per_sample


def initialize_coords(
    data: np.ndarray | scipy.sparse.coo_matrix | scipy.sparse.csr_matrix,
    graph: scipy.sparse.coo_matrix,
    n_components: int,
    init: Literal["random", "pca", "spectral"] | np.ndarray,
    random_state: int | RandomState | None,
    metric: str,
    metric_kwds: dict | None,
) -> np.ndarray:
    """Initialize the embedding: random, PCA, spectral or use a given array."""
    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "pca":
        if scipy.sparse.issparse(data):
            pca = TruncatedSVD(n_components=n_components, random_state=random_state)
        else:
            pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(data).astype(np.float32)
        embedding = noisy_scale_coords(
            embedding, random_state, max_coord=10, noise=0.0001
        )
    elif isinstance(init, str) and init == "spectral":
        embedding = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        # We add a little noise to avoid local minima for optimization to come
        embedding = noisy_scale_coords(
            embedding, random_state, max_coord=10, noise=0.0001
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            # 2D only: calculate nearest neighbors to find a good jitter scale
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, _ = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data
    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")
    return embedding


def noisy_scale_coords(
    coords: np.ndarray,
    random_state: RandomState,
    max_coord: float = 10.0,
    noise: float = 0.0001,
) -> np.ndarray:
    """Scale the coordinates to be between -max_coord and max_coord, and add noise."""
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(
        np.float32
    )


def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
    custom_epoch_func=_optimize_layout_euclidean_single_epoch,
    anneal_lr=True,
) -> tuple[np.ndarray, dict]:
    """Embed the UMAP graph into n_components dimensions using the UMAP algorithm."""
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    if n_epochs is None:
        n_epochs = default_epochs

    # If n_epoch is a list, get the maximum epoch to reach
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    embedding = initialize_coords(
        data, graph, n_components, init, random_state, metric, metric_kwds
    )

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    embedding = optimize_layout(
        embedding,
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
        move_other=True,
        custom_epoch_func=custom_epoch_func,
        anneal_lr=anneal_lr,
    )

    if isinstance(embedding, list):
        aux_data["embedding_list"] = embedding
        embedding = embedding[-1].copy()

    return embedding, aux_data


def optimize_layout(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
    move_other=False,
    custom_epoch_func=_optimize_layout_euclidean_single_epoch,
    anneal_lr=True,
) -> np.ndarray | list[np.ndarray]:
    """Optimize the low dimensional embedding using the given epochs and optimization
    function."""
    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    epoch_fn = numba.njit(custom_epoch_func, fastmath=True, parallel=parallel)

    if tqdm_kwds is None:
        tqdm_kwds = {}

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        epoch_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        if anneal_lr:
            alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(head_embedding.copy())

    # Add the last embedding to the list as well
    if epochs_list is not None:
        embedding_list.append(head_embedding.copy())

    return head_embedding if epochs_list is None else embedding_list


class CustomGradientUMAP(umap.UMAP):
    """Custom UMAP class that allows for custom gradient functions."""

    def __init__(self, custom_epoch_func, anneal_lr=True, **kwargs):
        super().__init__(**kwargs)
        self.custom_epoch_func = custom_epoch_func
        self.anneal_lr = anneal_lr

    def _fit_embed_data(
        self, X, n_epochs, init, random_state, **_
    ) -> tuple[np.ndarray, dict]:
        return simplicial_set_embedding(
            X,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self.random_state is None,
            self.verbose,
            self.tqdm_kwds,
            self.custom_epoch_func,
            self.anneal_lr,
        )
