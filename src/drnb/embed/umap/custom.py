import abc
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numba
import numpy as np
import scipy.sparse
import umap

# pylint: disable=no-name-in-module
from numpy.random import RandomState
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from umap.layouts import (
    clip,
    rdist,
    tau_rand_int,
)
from umap.spectral import spectral_layout
from umap.umap_ import INT32_MAX, INT32_MIN, make_epochs_per_sample

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


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
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    custom_epoch_func,
    custom_grad_coeff_attr,
    custom_grad_coeff_rep,
    grad_args,
    parallel: bool = False,
    verbose: bool = False,
    tqdm_kwds: dict | None = None,
    anneal_lr: bool = True,
) -> tuple[np.ndarray, dict]:
    """Embed data using a custom gradient function."""
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

    embedding = optimize_layout_euclidean(
        embedding,
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        rng_state,
        initial_alpha,
        negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
        move_other=True,
        custom_epoch_func=custom_epoch_func,
        custom_grad_coeff_attr=custom_grad_coeff_attr,
        custom_grad_coeff_rep=custom_grad_coeff_rep,
        grad_args=grad_args,
        anneal_lr=anneal_lr,
    )

    if isinstance(embedding, list):
        aux_data["embedding_list"] = embedding
        embedding = embedding[-1].copy()

    return embedding, aux_data


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    rng_state,
    initial_alpha,
    negative_sample_rate,
    parallel,
    verbose,
    tqdm_kwds,
    move_other,
    anneal_lr,
    custom_epoch_func,
    custom_grad_coeff_attr,
    custom_grad_coeff_rep,
    grad_args,
) -> np.ndarray | list[np.ndarray]:
    """Optimize the low dimensional embedding using a stochastic gradient descent
    method and the custom gradient functions."""

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    epoch_fn = numba.njit(custom_epoch_func, fastmath=True, parallel=parallel)
    grad_attr_fn = numba.njit(custom_grad_coeff_attr, fastmath=True)
    grap_rep_fn = numba.njit(custom_grad_coeff_rep, fastmath=True)

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
            rng_state,
            dim,
            move_other,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
            alpha,
            grad_attr_fn,
            grap_rep_fn,
            grad_args,
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


def epoch_func(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    rng_state,
    dim,
    move_other,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    iter,
    alpha,
    grad_coeff_attr,
    grad_coeff_rep,
    grad_args,
):
    """Perform a single epoch of optimization using the custom gradient functions."""
    # pylint: disable=not-an-iterable
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] > iter:
            continue

        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]

        dist_squared = rdist(current, other)

        n_neg_samples = int(
            (iter - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
        )

        if dist_squared > 0.0:
            grad_coeff = grad_coeff_attr(dist_squared, grad_args)
        else:
            grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))
            current[d] += grad_d * alpha
            if move_other:
                other[d] += -grad_d * alpha

        epoch_of_next_sample[i] += epochs_per_sample[i]

        for _ in range(n_neg_samples):
            k = tau_rand_int(rng_state) % n_vertices
            if j == k:
                continue
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = grad_coeff_rep(dist_squared, grad_args)
            else:
                grad_coeff = 0.0

            for d in range(dim):
                if grad_coeff > 0.0:
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                else:
                    grad_d = 4.0
                current[d] += grad_d * alpha

        epoch_of_next_negative_sample[i] += (
            n_neg_samples * epochs_per_negative_sample[i]
        )


class CustomGradientUMAP(umap.UMAP, abc.ABC):
    """Custom UMAP class that allows for custom gradient functions.

    Because this class extends `umap.UMAP`, implementing classes have access to all the
    attributes of `umap.UMAP`. Implementing classes must implement the
    `get_gradient_args` method, which should return a custom class (usually a NamedTuple)
    that contains the parameters needed for the custom gradient functions. Implementing
    classes must also implement the `custom_epoch_func`, `custom_attr_func`, and
    `custom_rep_func` methods, which should contain the custom gradient functions.
    """

    def __init__(
        self,
        custom_epoch_func,
        custom_attr_func,
        custom_rep_func,
        anneal_lr: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.custom_epoch_func = custom_epoch_func
        self.custom_attr_func = custom_attr_func
        self.custom_rep_func = custom_rep_func
        self.anneal_lr = anneal_lr

    @abc.abstractmethod
    def get_gradient_args(self) -> Any:
        """Return a custom class (usually a NamedTuple) that contains the parameters
        needed for the custom gradient functions."""

    def _fit_embed_data(self, X, n_epochs, init, random_state, **_):
        grad_args = self.get_gradient_args()
        return simplicial_set_embedding(
            X,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self.custom_epoch_func,
            self.custom_attr_func,
            self.custom_rep_func,
            grad_args,
            self.random_state is None,
            self.verbose,
            self.tqdm_kwds,
            self.anneal_lr,
        )


# Test implementation of UMAP
class UmapGradientArgs(NamedTuple):
    """Parameters for the custom UMAP gradient functions. This is a test implementation
    that replicates the default UMAP behavior."""

    a: float
    b: float
    gamma: float


def umap_grad_coeff_attr(dist_squared: float, grad_args: UmapGradientArgs) -> float:
    """Compute the gradient coefficient for the attractive force in UMAP."""
    a = grad_args.a
    b = grad_args.b
    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
    grad_coeff /= a * pow(dist_squared, b) + 1.0
    return grad_coeff


def umap_grad_coeff_rep(dist_squared: float, grad_args: UmapGradientArgs) -> float:
    """Compute the gradient coefficient for the repulsive force in UMAP."""
    a = grad_args.a
    b = grad_args.b
    gamma = grad_args.gamma
    grad_coeff = 2.0 * gamma * b
    grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return grad_coeff


class UMAP(CustomGradientUMAP):
    """Custom UMAP class that replicate the default UMAP behavior."""

    def get_gradient_args(self):
        return UmapGradientArgs(a=self._a, b=self._b, gamma=self.repulsion_strength)

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True
        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=umap_grad_coeff_attr,
            custom_rep_func=umap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Umap(drnb.embed.umap.Umap):
    """Embedder that implements default UMAP behavior via the CustomUmap class

    Attributes:
        use_precomputed_knn: bool - whether to use precomputed nearest neighbors
        drnb_init: str - method for initializing UMAP
    """

    use_precomputed_knn: bool = True
    drnb_init: str | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, UMAP, "UMAP")


@dataclass
class CustomUmap(drnb.embed.umap.Umap):
    """Embedding method that uses a custom UMAP class.

    Attributes:
        ctor: CustomGradientUMAP - custom UMAP class
        embedder_name: str - name of the custom UMAP class
        use_precomputed_knn: bool - whether to use precomputed nearest neighbors
        drnb_init: str - method for initializing UMAP
    """

    ctor: drnb.embed.umap.Umap | None = None
    embedder_name: str = "CustomUmap"
    use_precomputed_knn: bool = True
    drnb_init: str | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, self.ctor, self.embedder_name)


def custom_umap(ctor: Any, embedder_name: str = "CustomUmap") -> CustomUmap:
    """Adapts custom umap classes without having to add them to `embed.factory`.
    Pass `method=custom_umap(HTUMAP, "HT-UMAP")` instead of `method="htumap"` in
    e.g. `drnb.embed.pipeline.standard_eval`"""

    def factory(**kwargs):
        return CustomUmap(ctor=ctor, embedder_name=embedder_name, **kwargs)

    return factory
