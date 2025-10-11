#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pairs_from_knn(idx: np.ndarray, n_neighbors: int) -> np.ndarray:
    n = idx.shape[0]
    use = idx[:, 1 : 1 + n_neighbors] if idx.shape[1] > n_neighbors else idx[:, 1:]
    m = use.shape[1]
    pairs = np.empty((n * m, 2), dtype=np.int32)
    out = 0
    for i in range(n):
        nn = use[i]
        k = nn.shape[0]
        pairs[out : out + k, 0] = i
        pairs[out : out + k, 1] = nn
        out += k
    return pairs[:out]


def run_pacmap(req: dict) -> dict:
    import pacmap  # ensure installed in the environment

    _log("Running PaCMAP via plugin")

    x = np.load(req["input"]["x_path"])
    params = dict(req.get("params") or {})
    neigh = (req.get("input") or {}).get("neighbors") or {}
    idx_path = neigh.get("idx_path")
    pair_neighbors = None
    if idx_path:
        _log("using precomputed knn")
        idx = np.load(idx_path)
        n_neighbors = int(params.get("n_neighbors", 10))
        pair_neighbors = _pairs_from_knn(idx, n_neighbors)

    snaps = sorted(set((req.get("options") or {}).get("snapshots") or []))
    if snaps:
        params["intermediate"] = True
        params["intermediate_snapshots"] = snaps

    _log(f"PaCMAP params: {params}")
    emb = pacmap.PaCMAP(**params)
    result = emb.fit_transform(x, pair_neighbors=pair_neighbors)

    # Normalize to a single coords and optional snapshots
    save = {}
    if isinstance(result, np.ndarray) and result.ndim == 2:
        save["coords"] = result.astype(np.float32, copy=False)
    else:
        # result is a 3D array [N, 2, T] or list of (N, 2)
        if isinstance(result, np.ndarray) and result.ndim == 3:
            save["coords"] = result[:, :, -1].astype(np.float32, copy=False)
            for i, it in enumerate(snaps):
                save[f"snap_{it:04d}"] = result[:, :, i].astype(np.float32, copy=False)
        else:
            save["coords"] = result[-1].astype(np.float32, copy=False)
            for arr, it in zip(result, snaps):
                save[f"snap_{it:04d}"] = arr.astype(np.float32, copy=False)

    out = Path("result.npz").resolve()
    np.savez_compressed(out, **save)
    return {"ok": True, "result_npz": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method", required=True
    )  # "pacmap" (kept for symmetry if you add more)
    ap.add_argument("--request", required=True)  # path to JSON
    args = ap.parse_args()

    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        if args.method == "pacmap":
            resp = run_pacmap(req)
        else:
            resp = {"ok": False, "message": f"unknown method {args.method}"}
    except Exception as e:  # noqa: BLE001
        resp = {"ok": False, "message": str(e)}

    # IMPORTANT: exactly one JSON to stdout; logs go to stderr
    print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
