from dataclasses import dataclass

import numpy as np
import pynndescent
import scipy.sparse.csgraph
import umap
from sklearn.utils import check_random_state

import drnb.embed
import drnb.neighbors as nbrs
from drnb.log import log
from drnb.util import get_method_and_args
from drnb.yinit import gspectral, noisy_scale_coords, spca, tsvd_warm_spectral


def n_connected_components(graph):
    return scipy.sparse.csgraph.connected_components(graph)[0]


# A subclass of NNDescent which exists purely to escape the scrutiny of a validation
# type check in UMAP when using pre-computed knn.
# https://github.com/lmcinnes/umap/issues/848
class DummyNNDescent(pynndescent.NNDescent):
    # pylint: disable=super-init-not-called
    def __init__(self):
        return


def umap_knn(precomputed_knn):
    # we aren't going to transform new data so we don't need the search index
    # to actually work
    dummy_search_index = DummyNNDescent()
    return (
        precomputed_knn.idx,
        precomputed_knn.dist,
        dummy_search_index,
    )


def umap_spectral_init(
    x,
    knn=None,
    metric="euclidean",
    n_neighbors=15,
    random_state=42,
    tsvdw=False,
    tsvdw_tol=1e-5,
    jitter=True,
):
    if knn is None:
        nbr_data = nbrs.calculate_neighbors(
            x,
            n_neighbors=n_neighbors,
            metric=metric,
            method="pynndescent",
            return_distance=True,
            method_kwds=dict(random_state=random_state),
        )
        knn = [nbr_data.idx, nbr_data.dist]

    knn_fss = umap_graph(x, knn)

    nc = n_connected_components(knn_fss)
    if nc > 1:
        log.warning("UMAP graph has %d components", nc)

    random_state = check_random_state(random_state)

    if tsvdw:
        log.info("Using tsvdw solver")
        return tsvd_warm_spectral(
            graph=knn_fss,
            dim=2,
            random_state=random_state,
            tol=tsvdw_tol,
            jitter=jitter,
        )
    init = umap.umap_.spectral_layout(
        data=None,
        graph=knn_fss,
        dim=2,
        random_state=random_state,
    )
    if jitter:
        return noisy_scale_coords(init)
    return init


def umap_graph(x, knn):
    knn_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=knn[0],
        knn_dists=knn[1],
        n_neighbors=knn[0].shape[1],
        random_state=None,
        metric=None,
    )
    return knn_fss


@dataclass
class Umap(drnb.embed.Embedder):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def update_params(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = nbrs.get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )

            params["precomputed_knn"] = umap_knn(precomputed_knn)
            # also UMAP complains when a precomputed knn is used with a smaller dataset
            # unless this flag is set
            params["force_approximation_algorithm"] = True

        if self.drnb_init is not None:
            drnb_init, init_params = get_method_and_args(self.drnb_init, {})
            if drnb_init == "spca":
                params["init"] = spca(x)
            elif drnb_init == "global_spectral":
                params["init"] = gspectral(
                    x,
                    knn=params["precomputed_knn"],
                    op=init_params.get("op", "intersection"),
                    weight=init_params.get("weight", 0.2),
                    metric=params.get("metric", "euclidean"),
                    random_state=params.get("random_state", 42),
                    global_neighbors=init_params.get("global_neighbors", "random"),
                    spectral_algorithm=init_params.get("spectral_algorithm", "umap"),
                )
            elif drnb_init == "tsvd_spectral":
                log.info("Initializing via truncated SVD-warmed spectral")
                graph = umap_graph(x, params["precomputed_knn"])
                params["init"] = tsvd_warm_spectral(
                    graph,
                    dim=2,
                    random_state=params.get("random_state", 42),
                )
            else:
                raise ValueError(f"Unknown drnb initialization '{self.drnb_init}'")

        if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
            params["metric"] = "precomputed"

        # tqdm doesn't behave well in a notebook so if verbose is set, unless tqdm_kwds
        # has been set explicitly, disable it
        if params.get("verbose", False) and "tqdm_kwds" not in params:
            params["tqdm_kwds"] = {"disable": True}

        return params

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return embed_umap(x, params)


def embed_umap(
    x,
    params,
):
    log.info("Running UMAP")
    embedder = umap.UMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    if params.get("densmap", False) and params.get("output_dens", False):
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
