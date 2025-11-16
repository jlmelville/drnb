#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import openTSNE
from drnb.embed.tsne import get_tsne_affinities, tsne_annealed_exaggeration
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.results import write_response_json

_PLUGIN_ONLY_PARAMS = {
    "use_precomputed_knn",
    "affinity",
    "symmetrize",
    "n_neighbors",
    "anneal_exaggeration",
    "n_exaggeration_iter",
    "n_anneal_steps",
    "anneal_momentum",
    "initial_momentum",
    "final_momentum",
    "gradient_descent_params",
}


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != sdk_protocol.PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {sdk_protocol.PROTOCOL_VERSION}, got {proto}"
        )
    return data


def _load_initialization(req: dict[str, Any]) -> np.ndarray | str | None:
    init_path = (req.get("input") or {}).get("init_path")
    if init_path:
        return np.load(init_path, allow_pickle=False)
    params = req.get("params") or {}
    return params.get("initialization")


def _build_affinities(req: dict[str, Any], x: np.ndarray, ctx) -> Any:
    options = req.get("options") or {}
    if not options.get("use_precomputed_knn", True) or ctx is None:
        return None
    params = req.get("params") or {}
    return get_tsne_affinities(
        affinity_type=params.get("affinity", "perplexity"),
        perplexity=params.get("perplexity", 30.0),
        n_neighbors=params.get("n_neighbors"),
        x=x,
        metric=params.get("metric", "euclidean"),
        symmetrize=params.get("symmetrize", "mean"),
        ctx=ctx,
    )


def run_tsne(req: dict[str, Any]) -> dict[str, Any]:
    ctx = sdk_protocol.context_from_payload(req.get("context"))
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    init = _load_initialization(req)
    with redirect_stdout(sys.stderr):
        affinities = _build_affinities(req, x, ctx)

    anneal = bool(params.get("anneal_exaggeration", False))

    with redirect_stdout(sys.stderr):
        if anneal:
            if affinities is None:
                raise RuntimeError(
                    "Annealed exaggeration requires precomputed affinities"
                )
            early_exaggeration_iter = params.get("early_exaggeration_iter", 250)
            n_exaggeration_iter = int(early_exaggeration_iter / 2)
            n_anneal_steps = n_exaggeration_iter
            anneal_momentum = params.get(
                "anneal_momentum", params.get("final_momentum", 0.8)
            )
            gradient_descent_params = {
                "n_jobs": params.get("n_jobs", 1),
                "dof": params.get("dof", 1),
                "learning_rate": params.get("learning_rate", "auto"),
                "verbose": params.get("verbose", False),
                "theta": params.get("theta", 0.5),
                "n_interpolation_points": params.get("n_interpolation_points", 3),
                "min_num_intervals": params.get("min_num_intervals", 50),
                "ints_in_interval": params.get("ints_in_interval", 1),
                "max_grad_norm": params.get("max_grad_norm", None),
                "max_step_norm": params.get("max_step_norm", 5),
                "callbacks": params.get("callbacks", None),
                "callbacks_every_iters": params.get("callbacks_every_iters", 50),
            }
            embedded = tsne_annealed_exaggeration(
                data=x,
                affinities=affinities,
                random_state=params.get("random_state", 42),
                n_exaggeration_iter=n_exaggeration_iter,
                early_exaggeration=params.get("early_exaggeration", 12.0),
                initial_momentum=params.get("initial_momentum", 0.5),
                n_anneal_steps=n_anneal_steps,
                anneal_momentum=anneal_momentum,
                n_iter=params.get("n_iter", 500),
                final_momentum=params.get("final_momentum", 0.8),
                initialization=init,
                negative_gradient_method=params.get("negative_gradient_method", "auto"),
                gradient_descent_params=gradient_descent_params,
            )
        else:
            tsne_params = {
                key: value
                for key, value in params.items()
                if key not in _PLUGIN_ONLY_PARAMS
            }
            log(f"Running openTSNE.TSNE with params={summarize_params(tsne_params)}")
            tsne = openTSNE.TSNE(n_components=2, **tsne_params)
            embedded = tsne.fit(x, affinities=affinities, initialization=init)

    coords = np.asarray(embedded, dtype=np.float32, order="C")
    out_path = Path(req["output"]["result_path"]).resolve()
    np.savez_compressed(out_path, coords=coords)
    return {"ok": True, "result_npz": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req_path = Path(args.request)
    try:
        req = _load_request(req_path)
        if args.method != "tsne-plugin":
            raise RuntimeError(f"unknown method {args.method}")
        resp = run_tsne(req)
    except Exception as exc:  # noqa: BLE001
        resp = {"ok": False, "message": str(exc)}

    response_path = (req.get("output") or {}).get("response_path")
    if not response_path:
        raise RuntimeError("Request missing output.response_path")
    write_response_json(response_path, resp)
    log(f"Wrote response to {response_path}")


if __name__ == "__main__":
    main()
