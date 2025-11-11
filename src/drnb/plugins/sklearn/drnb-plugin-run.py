#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.random_projection import SparseRandomProjection

from drnb.plugins.protocol import PROTOCOL_VERSION, context_from_payload


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {PROTOCOL_VERSION}, got {proto}"
        )
    return data


@dataclass(frozen=True)
class SklearnMethod:
    ctor: Callable[..., Any]
    label: str
    enforce: dict[str, Any] | None = None

    def build(self, n_components: int, params: dict[str, Any]) -> Any:
        kwargs = dict(params or {})
        kwargs.setdefault("n_components", n_components)
        if self.enforce:
            kwargs.update(self.enforce)
        return self.ctor(**kwargs)


METHODS: dict[str, SklearnMethod] = {
    "pca-plugin": SklearnMethod(PCA, "PCA"),
    "randproj-plugin": SklearnMethod(
        SparseRandomProjection, "Sparse Random Projection"
    ),
    "isomap-plugin": SklearnMethod(Isomap, "Isomap"),
    "mmds-plugin": SklearnMethod(
        MDS,
        "Sklearn MMDS",
        enforce={"metric": True, "normalized_stress": False, "n_init": 1},
    ),
    "nmds-plugin": SklearnMethod(
        MDS,
        "Sklearn NMDS",
        enforce={"metric": False, "normalized_stress": False, "n_init": 1},
    ),
}


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise RuntimeError(f"unknown method {method}")

    # Ensure EmbedContext deserializes even if unused for now
    context_from_payload(req.get("context"))

    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})
    info = METHODS[method]

    requested_components = params.pop("n_components", None)
    n_components = 2
    if requested_components is not None and int(requested_components) != 2:
        _log(
            f"{info.label}: ignoring n_components={requested_components}; plugin outputs 2D embeddings."
        )

    estimator = info.build(n_components=n_components, params=params)

    _log(f"Running {info.label} with params={params} n_components={n_components}")
    coords = estimator.fit_transform(x).astype(np.float32, copy=False)

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
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _log(tb)
        resp = {"ok": False, "message": tb or str(exc)}

    print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
