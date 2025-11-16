#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pacmap
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.results import write_response_json

from drnb.embed.context import get_neighbors_with_ctx
from drnb.embed.deprecated.pacmap import create_neighbor_pairs
from drnb.neighbors.localscale import locally_scaled_neighbors


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != sdk_protocol.PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {sdk_protocol.PROTOCOL_VERSION}, got {proto}"
        )
    return data


def _load_array(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    return np.load(path, allow_pickle=False)


def _load_initialization(req: dict[str, Any], params: dict[str, Any]) -> Any:
    init = params.pop("init", None)
    init_path = (req.get("input") or {}).get("init_path")
    array = _load_array(init_path)
    if array is not None:
        return array
    return init


def _needs_precomputed(
    options: dict[str, Any], params: dict[str, Any], x: np.ndarray
) -> bool:
    if not options.get("use_precomputed_knn", True):
        return False
    apply_pca = params.get("apply_pca", True)
    if apply_pca and x.shape[1] > 100:
        log("Precomputed knn cannot be used: dimensionality will be reduced via PCA")
        return False
    return True


def _load_neighbors(
    req: dict[str, Any],
    ctx,
    metric: str,
    n_neighbors: int,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    neigh = (req.get("input") or {}).get("neighbors") or {}
    idx = _load_array(neigh.get("idx_path"))
    dist = _load_array(neigh.get("dist_path"))
    if idx is not None and idx.shape[1] >= n_neighbors:
        if dist is None:
            dist = np.zeros_like(idx, dtype=np.float32)
        return idx, dist
    if ctx is None:
        return (None, None)
    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
    if pre is None:
        return (None, None)
    dist = pre.dist if pre.dist is not None else np.zeros_like(pre.idx)
    return pre.idx, dist


def _prepare_pair_neighbors(
    req: dict[str, Any],
    params: dict[str, Any],
    x: np.ndarray,
    ctx,
    *,
    local_scale: bool,
    local_scale_kwargs: dict[str, Any] | None,
) -> np.ndarray | None:
    if not _needs_precomputed(req.get("options") or {}, params, x):
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
    idx, dist = _load_neighbors(req, ctx, metric, knn_neighbors, x)
    if idx is None or dist is None:
        log("No precomputed knn available; plugin will rely on PaCMAP defaults")
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
    pair_neighbors = create_neighbor_pairs(idx, neighbor_count)
    return pair_neighbors


def _configure_snapshots(params: dict[str, Any]) -> list[int]:
    snaps = params.get("intermediate_snapshots") or []
    if snaps:
        params["intermediate"] = True
        # PaCMAP uses "num_iters" to specify the number of iterations per phase
        # so let's be super careful with naming to not confuse ourselves
        num_iters_per_phase: tuple[int, ...] = params.get("num_iters", (450,))
        total_iters = sum(num_iters_per_phase)
        if snaps[-1] != total_iters:
            snaps.append(total_iters)
        params["intermediate_snapshots"] = snaps
    return snaps


def _extract_snapshot_arrays(
    result: Any, snapshots: list[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if isinstance(result, dict) and "coords" in result:
        coords = np.asarray(result["coords"])
        snap_map: dict[int, np.ndarray] = {}
        for key, value in (result.get("snapshots") or {}).items():
            try:
                iteration = int(str(key).split("_")[1])
            except (IndexError, ValueError):
                continue
            snap_map[iteration] = np.asarray(value)
        return coords, snap_map

    array = np.asarray(result)
    if array.ndim == 2:
        return array, {}

    if not snapshots:
        # No metadata to align snapshots; treat trailing slice as coords only.
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
    snap_map: dict[int, np.ndarray] = {}
    limit = min(expected, series.shape[0])
    for i in range(limit):
        snap_map[snapshots[i]] = series[i]
    return coords, snap_map


def _save_result(result, snapshots: list[int], out_path: Path) -> dict[str, Any]:
    coords, snap_map = _extract_snapshot_arrays(result, snapshots)
    save: dict[str, np.ndarray] = {"coords": coords.astype(np.float32, copy=False)}
    for iteration, array in snap_map.items():
        save[f"snap_{iteration}"] = array.astype(np.float32, copy=False)
    np.savez_compressed(out_path, **save)
    return {"ok": True, "result_npz": str(out_path)}


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    ctx = sdk_protocol.context_from_payload(req.get("context"))
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    options = req.get("options") or {}
    init = _load_initialization(req, params)
    snapshots = _configure_snapshots(params)

    local_scale = params.pop("local_scale", True)
    local_scale_kwargs = params.pop("local_scale_kwargs", None)

    pair_neighbors = _prepare_pair_neighbors(
        req,
        params,
        x,
        ctx,
        local_scale=local_scale,
        local_scale_kwargs=local_scale_kwargs,
    )
    if pair_neighbors is not None:
        params["pair_neighbors"] = pair_neighbors

    summarized = summarize_params(params)
    if method == "pacmap-plugin":
        log(f"Running PaCMAP with params={summarized}")
        embedder = pacmap.PaCMAP(**params)
    elif method == "localmap-plugin":
        log(f"Running LocalMAP with params={summarized}")
        embedder = pacmap.LocalMAP(**params)
    else:
        raise RuntimeError(f"unknown method {method}")

    result = embedder.fit_transform(x, init=init)
    out_path = Path(req["output"]["result_path"]).resolve()
    return _save_result(result, snapshots, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req = _load_request(Path(args.request))
    try:
        resp = run_method(req, args.method)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        resp = {"ok": False, "message": tb or str(exc)}

    response_path = (req.get("output") or {}).get("response_path")
    if not response_path:
        raise RuntimeError("Request missing output.response_path")
    write_response_json(response_path, resp)
    log(f"Wrote response to {response_path}")


if __name__ == "__main__":
    main()
