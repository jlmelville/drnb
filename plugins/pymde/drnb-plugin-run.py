#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymde
import scipy.sparse as sp
import torch
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log
from drnb_plugin_sdk.helpers.results import write_response_json
from pymde import constraints, preprocess, problem, quadratic
from pymde.functions import penalties
from pymde.preprocess.graph import Graph
from pymde.recipes import _remove_anchor_anchor_edges

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.embed.deprecated.pymde import (
    embed_pymde_nbrs,
    nn_to_graph,
    pymde_n_neighbors,
)
from drnb.log import log
from drnb.neighbors import NearestNeighbors
from drnb.types import EmbedResult
from drnb.yinit import spca


@dataclass
class Pymde(drnb.embed.base.Embedder):
    """PyMDE embedder.

    Attributes:
        use_precomputed_knn: Whether to use precomputed knn.
        drnb_init: DRNB initialization method, one of "spca" or None.
        seed: Random seed.
    """

    use_precomputed_knn: bool = True
    drnb_init: str = None
    seed: int = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
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
            precomputed_knn = get_neighbors_with_ctx(
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
def pymde_n_neighbors(n: int) -> int:
    """Calculate the default number of neighbors for PyMDE (between 5 and 15)."""
    n_choose_2 = n * (n - 1) / 2
    return int(max(min(15, n_choose_2 * 0.01 / n), 5))


def embed_pymde_nbrs(
    x: np.ndarray, seed: int, params: dict, graph: Graph | None = None
) -> np.ndarray:
    """Embed using PyMDE with nearest neighbors."""
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)

    if graph is not None:
        log.info("Running preserve neighbors with knn")
        embedder = _preserve_neighbors_knn(
            graph, device=device, embedding_dim=2, **params
        )
    else:
        log.info("Running PyMDE preserve neighbors")
        embedder = pymde.preserve_neighbors(x, device=device, embedding_dim=2, **params)
    embedded = embedder.embed().cpu().data.numpy()
    log.info("Embedding completed")

    return embedded


# this code has been stitched together from various parts of pymde.preprocess
def _idx_to_edgelist(idx: np.ndarray, canonicalize_edges: bool = True) -> np.ndarray:
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


def _edgelist_to_sparse(edgelist: np.ndarray) -> sp.csr_matrix:
    n_items = edgelist.max() + 1
    rows = edgelist[:, 0]
    cols = edgelist[:, 1]
    distances = np.ones(edgelist.shape[0], dtype=np.float32)
    graph = sp.coo_matrix((distances, (rows, cols)), shape=(n_items, n_items))
    graph = graph + graph.T
    return graph.tocsr()


def nn_to_graph(nn: NearestNeighbors) -> Graph:
    """Convert NearestNeighbors to Graph."""
    # ignore self neighbor
    idx = nn.idx[:, 1:]
    edgelist = _idx_to_edgelist(idx)
    csr = _edgelist_to_sparse(edgelist)
    return Graph(csr)


# hacked version of pymde.preserve_neighbors to allow a pre-calculated Graph
def _preserve_neighbors_knn(
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


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != sdk_protocol.PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {sdk_protocol.PROTOCOL_VERSION}, got {proto}"
        )
    return data


def _load_init(req: dict[str, Any], params: dict[str, Any]) -> None:
    init_path = (req.get("input") or {}).get("init_path")
    if init_path:
        params["init"] = np.load(init_path, allow_pickle=False)


def _build_graph(
    req: dict[str, Any],
    params: dict[str, Any],
    x: np.ndarray,
    ctx,
) -> Any | None:
    options = req.get("options") or {}
    use_knn = options.get("use_precomputed_knn")
    if use_knn is None:
        use_knn = True
    if not use_knn:
        return None

    if ctx is None:
        log("No EmbedContext supplied; PyMDE plugin cannot reuse neighbors")
        return None

    metric = params.get("distance", "euclidean")
    default_neighbors = pymde_n_neighbors(x.shape[0])
    n_neighbors = int(params.get("n_neighbors", default_neighbors)) + 1

    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
    if pre is None:
        log("No precomputed knn available; PyMDE will use its internal graph")
        return None
    return nn_to_graph(pre)


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    if method != "pymde-plugin":
        raise RuntimeError(f"unknown method {method}")

    ctx = sdk_protocol.context_from_payload(req.get("context"))
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    _load_init(req, params)
    graph = _build_graph(req, params, x, ctx)

    seed = params.pop("seed", None)

    log(f"Running PyMDE with params={params}")
    coords = embed_pymde_nbrs(x, seed=seed, params=params, graph=graph).astype(
        np.float32, copy=False
    )

    result_path = Path(req["output"]["result_path"]).resolve()
    np.savez_compressed(result_path, coords=coords)
    return {"ok": True, "result_npz": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req = _load_request(Path(args.request))
    try:
        resp = run_method(req, args.method)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        resp = {"ok": False, "message": tb}

    response_path = (req.get("output") or {}).get("response_path")
    if not response_path:
        raise RuntimeError("Request missing output.response_path")
    write_response_json(response_path, resp)
    log(f"Wrote response to {response_path}")


if __name__ == "__main__":
    main()
