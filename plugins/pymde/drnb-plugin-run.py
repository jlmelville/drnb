#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import numpy as np
import pymde
import scipy.sparse as sp
import torch
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.paths import (
    resolve_init_path,
    resolve_neighbors,
    resolve_x_path,
)
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin
from drnb_plugin_sdk.helpers.version import build_version_payload
from pymde import constraints, preprocess, problem, quadratic
from pymde.functions import penalties
from pymde.preprocess.graph import Graph
from pymde.recipes import _remove_anchor_anchor_edges

VERSION_INFO = build_version_payload(package="pymde")


def embed_pymde_nbrs(
    x: np.ndarray, seed: int, params: dict, graph: Graph | None = None
) -> np.ndarray:
    """Embed using PyMDE with nearest neighbors."""
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)

    if graph is not None:
        log("Running preserve neighbors with knn")
        embedder = _preserve_neighbors_knn(
            graph, device=device, embedding_dim=2, **params
        )
    else:
        log("Running PyMDE preserve neighbors")
        embedder = pymde.preserve_neighbors(x, device=device, embedding_dim=2, **params)
    embedded = embedder.embed().cpu().data.numpy()
    log("Embedding completed")

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


def nn_to_graph(idx: np.ndarray) -> Graph:
    """Convert neighbor index array to Graph."""
    # ignore self neighbor
    idx = idx[:, 1:]
    edgelist = _idx_to_edgelist(idx)
    csr = _edgelist_to_sparse(edgelist)
    return Graph(csr)


# Penalties that pymde explicitly recommends for repulsive (negative) weights.
_ALLOWED_REPULSIVE = {"Log", "InvPower", "LogRatio"}


def _resolve_penalty(value: Any, *, kind: str) -> Any:
    """
    Resolve a penalty specification to a pymde.functions.penalties class.

    Rules:
      - None        -> None
      - callable    -> returned unchanged
      - string name -> looked up in pymde.functions.penalties, case-insensitive

    Examples:
      "Quadratic"  -> penalties.Quadratic
      "quadratic"  -> penalties.Quadratic
      "LOG1P"      -> penalties.Log1p
      "logratio"   -> penalties.LogRatio
    """
    if value is None:
        return None

    # If the user passed a class/callable already, leave it as-is.
    if not isinstance(value, str):
        return value

    name = value
    attr_name: str | None = None

    # 1) Try direct attribute access first (exact name)
    if hasattr(penalties, name):
        attr_name = name
    else:
        # 2) Fallback: case-insensitive match against attributes in penalties
        target = name.lower()
        for candidate in dir(penalties):
            if candidate.lower() == target:
                attr_name = candidate
                break

    if attr_name is None:
        raise ValueError(
            f"Unknown {kind} '{value}'. "
            "It must be the name (case-insensitive) of a class in "
            "pymde.functions.penalties."
        )

    penalty_cls = getattr(penalties, attr_name)

    # Optional safety: for repulsive penalties, only allow the three
    # recommended ones from the pymde docstring.
    if kind == "repulsive_penalty":
        if attr_name.lower() not in _ALLOWED_REPULSIVE:
            allowed_str = ", ".join(sorted(_ALLOWED_REPULSIVE))
            raise ValueError(
                f"Repulsive penalty '{value}' is not recommended. "
                f"Use one of: {allowed_str}"
            )

    return penalty_cls


def _normalize_penalty_params(params: dict[str, Any]) -> None:
    """
    In-place conversion of string penalty names in `params` to actual classes.

    Expected strings are exact class names in pymde.functions.penalties,
    e.g. 'Quadratic', 'Log1p', 'Log', 'InvPower', 'LogRatio'.
    """
    if "attractive_penalty" in params:
        attractive_penalty = params["attractive_penalty"]
        if isinstance(attractive_penalty, str):
            log(f"attractive penalty: {attractive_penalty}")
        attractive_penalty = _resolve_penalty(
            attractive_penalty,
            kind="attractive_penalty",
        )
        params["attractive_penalty"] = attractive_penalty

    if "repulsive_penalty" in params:
        repulsive_penalty = params["repulsive_penalty"]
        if isinstance(repulsive_penalty, str):
            log(f"repulsive penalty: {repulsive_penalty}")
        params["repulsive_penalty"] = _resolve_penalty(
            repulsive_penalty,
            kind="repulsive_penalty",
        )


# Simple mapping from user-friendly names to zero-arg factories.
_CONSTRAINT_FACTORIES: dict[str, callable] = {
    "centered": constraints.Centered,
    "standardized": constraints.Standardized,
}


def _resolve_constraint(value: Any) -> Any:
    """
    Resolve a constraint specification to a pymde.constraints.Constraint.

    Allowed forms:
      - None                    -> None
      - instance of Constraint  -> returned unchanged
      - string                  -> 'centered' / 'standardized' (case-insensitive)

    Examples:
      "centered"     -> constraints.Centered()
      "Centered"     -> constraints.Centered()
      "STANDARDIZED" -> constraints.Standardized()
    """
    if value is None:
        return None

    # Already a concrete constraint instance (or some custom object) – leave it.
    if isinstance(value, constraints.Constraint):
        return value

    if not isinstance(value, str):
        # e.g. someone passed a custom Constraint subclass instance or factory;
        # don't get clever, just pass it through.
        return value

    key = value.strip().lower()

    # Optional: treat explicit "none" as no constraint.
    if key in {"none", "no", "null"}:
        return None

    try:
        factory = _CONSTRAINT_FACTORIES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_CONSTRAINT_FACTORIES.keys()))
        raise ValueError(
            f"Unknown constraint '{value}'. Valid constraint names are: {valid}"
        ) from exc

    # Call the zero-arg factory to get the singleton constraint object.
    return factory()


def _normalize_constraint_param(params: dict[str, Any]) -> None:
    """
    In-place conversion of a string 'constraint' param to a Constraint instance.

    If the caller doesn't provide 'constraint', we leave defaulting to
    _preserve_neighbors_knn / pymde.preserve_neighbors.
    """
    if "constraint" in params:
        constraint = params["constraint"]
        if isinstance(constraint, str):
            log(f"constraint: {constraint}")
        params["constraint"] = _resolve_constraint(constraint)


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


def _load_init(req: sdk_protocol.PluginRequest, params: dict[str, Any]) -> None:
    init_path = resolve_init_path(req)
    if init_path:
        params["init"] = np.load(init_path, allow_pickle=False)


def _build_graph(req: sdk_protocol.PluginRequest) -> Graph | None:
    if not req.options.use_precomputed_knn:
        return None

    neighbors = resolve_neighbors(req)
    if neighbors is None or not neighbors.idx_path:
        log("No precomputed knn available; PyMDE will use its internal graph")
        return None

    try:
        idx = np.load(neighbors.idx_path, allow_pickle=False).astype(
            np.int64, copy=False
        )
    except Exception as exc:  # noqa: BLE001
        log(f"Failed to load precomputed knn: {exc}")
        return None
    return nn_to_graph(idx)


def run_pymde(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    _load_init(req, params)
    _normalize_penalty_params(params)
    _normalize_constraint_param(params)
    graph = _build_graph(req)

    seed = params.pop("seed", None)

    log(f"Running PyMDE with params={summarize_params(params)}")
    coords = embed_pymde_nbrs(x, seed=seed, params=params, graph=graph).astype(
        np.float32, copy=False
    )

    return save_result_npz(req.output.result_path, coords, version=VERSION_INFO)


if __name__ == "__main__":
    run_plugin({"pymde": run_pymde})
