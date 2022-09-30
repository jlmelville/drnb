from dataclasses import dataclass

import numpy as np
import pymde
import scipy.sparse as sp
import torch
from pymde import constraints, preprocess, problem, quadratic
from pymde.functions import penalties
from pymde.preprocess.graph import Graph
from pymde.recipes import _remove_anchor_anchor_edges

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log
from drnb.yinit import spca


@dataclass
class Pymde(drnb.embed.Embedder):
    use_precomputed_knn: bool = True
    drnb_init: str = None
    seed: int = None

    def embed_impl(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        graph = None
        if self.use_precomputed_knn:
            metric = params.get("distance", "euclidean")
            default_n_neighbors = pymde_n_neighbors(x.shape[0])
            n_neighbors = params.get("n_neighbors", default_n_neighbors) + 1
            log.info("Using precomputed knn with n_neighbors = %d", n_neighbors)
            precomputed_knn = knn.get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )
            graph = nn_to_graph(precomputed_knn)

        if self.drnb_init is not None:
            if not self.use_precomputed_knn:
                raise ValueError("Must use precomputed knn with drnb_init")
            if self.drnb_init == "spca":
                params["init"] = spca(x)
            else:
                raise ValueError(f"Unknown drnb initialization '{self.drnb_init}'")

        return embed_pymde_nbrs(x, self.seed, params, graph=graph)


# from preserve_neighbors
def pymde_n_neighbors(n):
    n_choose_2 = n * (n - 1) / 2
    return int(max(min(15, n_choose_2 * 0.01 / n), 5))


def embed_pymde_nbrs(x, seed, params, graph=None):
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)

    if graph is not None:
        log.info("Running preserve neighbors with knn")
        embedder = preserve_neighbors_knn(
            graph, device=device, embedding_dim=2, **params
        )
    else:
        log.info("Running PyMDE preserve neighbors")
        embedder = pymde.preserve_neighbors(x, device=device, embedding_dim=2, **params)
    embedded = embedder.embed().cpu().data.numpy()
    log.info("Embedding completed")

    return embedded


# this code has been stitched together from various parts of pymde.preprocess
def idx_to_edgelist(idx, canonicalize_edges=True):
    n = idx.shape[0]
    n_neighbors = idx.shape[1]
    items = np.arange(n)
    items = np.repeat(items, n_neighbors)
    edges = np.stack([items, idx.flatten()], axis=1)

    if canonicalize_edges:
        flip_idx = edges[:, 0] > edges[:, 1]
        edges[flip_idx] = np.stack(
            [edges[flip_idx][:, 1], edges[flip_idx][:, 0]], axis=1
        )
    return edges


def edgelist_to_sparse(edgelist):
    n_items = edgelist.max() + 1
    rows = edgelist[:, 0]
    cols = edgelist[:, 1]
    distances = np.ones(edgelist.shape[0], dtype=np.float32)
    graph = sp.coo_matrix((distances, (rows, cols)), shape=(n_items, n_items))
    graph = graph + graph.T
    return graph.tocsr()


def idx_to_sparse(idx):
    edgelist = idx_to_edgelist(idx)
    return edgelist_to_sparse(edgelist)


def idx_to_graph(idx):
    csr = idx_to_sparse(idx)
    return Graph(csr)


def nn_to_graph(nn):
    # ignore self neighbor
    return idx_to_graph(nn.idx[:, 1:])


# hacked version of pymde.preserve_neighbors to allow a pre-calculated Graph
def preserve_neighbors_knn(
    knn_graph,
    embedding_dim=2,
    attractive_penalty=penalties.Log1p,
    repulsive_penalty=penalties.Log,
    constraint=None,
    repulsive_fraction=None,
    init="quadratic",
    device="cpu",
):
    n = knn_graph.n_items
    if constraint is None and repulsive_penalty is not None:
        constraint = constraints.Centered()
    elif constraint is None and repulsive_penalty is None:
        constraint = constraints.Standardized()

    edges = knn_graph.edges.to(device)
    weights = knn_graph.weights.to(device)

    if isinstance(constraint, constraints.Anchored):
        # remove anchor-anchor edges before generating intialization
        edges, weights = _remove_anchor_anchor_edges(edges, weights, constraint.anchors)

    if isinstance(init, np.ndarray):
        if init.shape != (n, embedding_dim):
            raise ValueError("Init matrix has wrong shape.")
        X_init = torch.tensor(init, device=device, dtype=torch.float)
    elif init == "quadratic":
        # use cg + torch when using GPU
        cg = device == "cuda"
        X_init = quadratic.spectral(
            n,
            embedding_dim,
            edges,
            weights,
            max_iter=1000,
            device=device,
            cg=cg,
        )
        # pylint: disable=protected-access
        if not isinstance(
            constraint, (constraints._Centered, constraints._Standardized)
        ):
            constraint.project_onto_constraint(X_init, inplace=True)
    elif init == "random":
        X_init = constraint.initialization(n, embedding_dim, device)
    else:
        raise ValueError(
            f"Unsupported value '{init}' for keyword argument `init`; "
            "the supported values are 'quadratic' and 'random'."
        )

    if repulsive_penalty is not None:
        if repulsive_fraction is None:
            # pylint: disable=protected-access
            if isinstance(constraint, constraints._Standardized):
                # standardization constraint already implicity spreads,
                # so use a lower replusion
                repulsive_fraction = 0.5
            else:
                repulsive_fraction = 1

        n_choose_2 = int(n * (n - 1) / 2)
        n_repulsive = int(repulsive_fraction * (edges.shape[0]))
        # cannot sample more edges than there are available
        n_repulsive = min(n_repulsive, n_choose_2 - edges.shape[0])

        negative_edges = preprocess.sample_edges(n, n_repulsive, exclude=edges).to(
            device
        )

        negative_weights = -torch.ones(
            negative_edges.shape[0], dtype=X_init.dtype, device=device
        )

        if isinstance(constraint, constraints.Anchored):
            negative_edges, negative_weights = _remove_anchor_anchor_edges(
                negative_edges, negative_weights, constraint.anchors
            )

        edges = torch.cat([edges, negative_edges])
        weights = torch.cat([weights, negative_weights])

        f = penalties.PushAndPull(
            weights,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
        )
    else:
        f = attractive_penalty(weights)

    mde = problem.MDE(
        n_items=n,
        embedding_dim=embedding_dim,
        edges=edges,
        distortion_function=f,
        constraint=constraint,
        device=device,
    )
    # pylint: disable=protected-access
    mde._X_init = X_init

    distances = mde.distances(mde._X_init)
    if (distances == 0).any():
        # pathological scenario in which at least two points overlap can yield
        # non-differentiable average distortion. perturb the initialization to
        # mitigate.
        mde._X_init += 1e-4 * torch.randn(
            mde._X_init.shape,
            device=mde._X_init.device,
            dtype=mde._X_init.dtype,
        )
    return mde
