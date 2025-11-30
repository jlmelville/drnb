#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import numpy as np
import pacmap
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
from localscale import locally_scaled_neighbors
from neighbor_pairs import create_neighbor_pairs

VERSION_INFO = build_version_payload(package="pacmap")


def _load_initialization(
    req: sdk_protocol.PluginRequest, params: dict[str, Any]
) -> Any:
    init = params.pop("init", None)
    init_path = resolve_init_path(req)
    if init_path:
        array = np.load(init_path, allow_pickle=False)
        return array
    return init


def _needs_precomputed(
    use_precomputed_knn: bool, params: dict[str, Any], x: np.ndarray
) -> bool:
    if not use_precomputed_knn:
        return False
    apply_pca = params.get("apply_pca", True)
    if apply_pca and x.shape[1] > 100:
        log("Precomputed knn cannot be used: dimensionality will be reduced via PCA")
        return False
    return True


def _load_neighbors(
    neighbors: sdk_protocol.PluginNeighbors, metric: str, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if neighbors and neighbors.idx_path:
        try:
            idx = np.load(neighbors.idx_path, allow_pickle=False)
            dist = (
                np.load(neighbors.dist_path, allow_pickle=False)
                if neighbors.dist_path
                else np.zeros_like(idx, dtype=np.float32)
            )
            return idx, dist
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to load precomputed knn: {exc}")
    log(f"No precomputed knn available for metric={metric}, n_neighbors={n_neighbors}")
    return (None, None)


def _prepare_pair_neighbors(
    req: sdk_protocol.PluginRequest,
    params: dict[str, Any],
    x: np.ndarray,
    neighbors: sdk_protocol.PluginNeighbors,
    *,
    local_scale: bool,
    local_scale_kwargs: dict[str, Any] | None,
) -> np.ndarray | None:
    use_knn = req.options.use_precomputed_knn
    if use_knn is None:
        use_knn = True
    if not _needs_precomputed(use_knn, params, x):
        return None
    metric = params.get("distance", params.get("metric", "euclidean"))
    base_neighbors = int(params.get("n_neighbors", 10)) + 1
    scale_kwargs = {
        "l": base_neighbors,
        "m": base_neighbors + 50,
        "scale_from": 4,
        "scale_to": 6,
    }
    if local_scale_kwargs:
        scale_kwargs.update(local_scale_kwargs)
    knn_neighbors = scale_kwargs["m"] if local_scale else base_neighbors
    idx, dist = _load_neighbors(neighbors, metric, knn_neighbors)
    if idx is None or dist is None or idx.shape[1] < base_neighbors:
        log("No usable precomputed knn available; plugin will rely on PaCMAP defaults")
        return None
    if local_scale:
        max_cols = idx.shape[1]
        scale_kwargs["m"] = min(scale_kwargs["m"], max_cols)
        scale_kwargs["l"] = min(scale_kwargs["l"], scale_kwargs["m"])
        idx, _ = locally_scaled_neighbors(idx, dist, **scale_kwargs)
    else:
        use_cols = min(base_neighbors, idx.shape[1])
        idx = idx[:, :use_cols]
    neighbor_count = min(base_neighbors - 1, idx.shape[1] - 1)
    if neighbor_count <= 0:
        return None
    return create_neighbor_pairs(idx, neighbor_count)


def _configure_snapshots(params: dict[str, Any]) -> list[int]:
    snaps = params.get("intermediate_snapshots") or []
    if snaps:
        params["intermediate"] = True
        num_iters_per_phase: tuple[int, ...] = params.get("num_iters", (450,))
        total_iters = sum(num_iters_per_phase)
        if snaps[-1] != total_iters:
            snaps.append(total_iters)
        params["intermediate_snapshots"] = snaps
    return snaps


def _extract_snapshot_arrays(
    result: Any, snapshots: list[int]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if isinstance(result, dict) and "coords" in result:
        coords = np.asarray(result["coords"])
        snap_map: dict[str, np.ndarray] = {}
        for key, value in (result.get("snapshots") or {}).items():
            try:
                iteration = int(str(key).split("_")[1])
            except (IndexError, ValueError):
                continue
            snap_map[f"snap_{iteration}"] = np.asarray(value)
        return coords, snap_map

    array = np.asarray(result)
    if array.ndim == 2:
        return array, {}

    if not snapshots:
        return array[-1], {}

    series = array
    expected = len(snapshots)
    if (
        series.ndim >= 3
        and series.shape[0] != expected
        and series.shape[-1] == expected
    ):
        series = np.moveaxis(series, -1, 0)

    coords = series[-1]
    snap_map: dict[str, np.ndarray] = {}
    limit = min(expected, series.shape[0])
    for i in range(limit):
        snap_map[f"snap_{snapshots[i]}"] = series[i]
    return coords, snap_map


def run_pacmap(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})
    init = _load_initialization(req, params)
    snapshots = _configure_snapshots(params)
    neighbors = resolve_neighbors(req)

    local_scale = params.pop("local_scale", True)
    local_scale_kwargs = params.pop("local_scale_kwargs", None)

    pair_neighbors = _prepare_pair_neighbors(
        req,
        params,
        x,
        neighbors,
        local_scale=local_scale,
        local_scale_kwargs=local_scale_kwargs,
    )
    if pair_neighbors is not None:
        params["pair_neighbors"] = pair_neighbors

    summarized = summarize_params(params)
    log(f"Running PaCMAP with params={summarized}")
    embedder = pacmap.PaCMAP(**params)

    result = embedder.fit_transform(x, init=init)
    coords, snaps = _extract_snapshot_arrays(result, snapshots)
    return save_result_npz(
        req.output.result_path, coords, snapshots=snaps, version=VERSION_INFO
    )


def run_localmap(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    init = _load_initialization(req, params)
    snapshots = _configure_snapshots(params)
    neighbors = resolve_neighbors(req)
    params.pop("local_scale", None)
    params.pop("local_scale_kwargs", None)

    pair_neighbors = _prepare_pair_neighbors(
        req,
        params,
        x,
        neighbors,
        local_scale=False,
        local_scale_kwargs=None,
    )
    if pair_neighbors is not None:
        params["pair_neighbors"] = pair_neighbors

    summarized = summarize_params(params)
    log(f"Running LocalMAP with params={summarized}")
    embedder = pacmap.LocalMAP(**params)

    result = embedder.fit_transform(x, init=init)
    coords, snaps = _extract_snapshot_arrays(result, snapshots)
    return save_result_npz(
        req.output.result_path, coords, snapshots=snaps, version=VERSION_INFO
    )


if __name__ == "__main__":
    run_plugin({"pacmap": run_pacmap, "localmap": run_localmap})
