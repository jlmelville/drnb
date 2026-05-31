#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import numpy as np
import topo as tp
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.paths import (
    resolve_x_path,
)
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin
from drnb_plugin_sdk.helpers.version import build_version_payload

VERSION_INFO = build_version_payload(package="topometry")


def run_topometry(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {}) | {
        "projection_methods": None,
        "cache": False,
        "multiscale": True,
    }

    if "min_eigs" not in params:
        # TopOMetry 1.1 requires this to be strictly below min(X.shape).
        min_eigs = max(1, min(min(x.shape) - 1, 128))
        log(f"Setting min_eigs to {min_eigs}")
        params["min_eigs"] = min_eigs

    projection_method = params.pop("projection_method", "MAP")
    multiscale = params.pop("multiscale", False)

    log(
        f"Running topometry with params={summarize_params(params)}"
        + f", projection method={projection_method}"
        + f", multiscale={multiscale}"
    )

    embedder = tp.TopOGraph(**params)
    embedder.fit(x)
    result = embedder.project(
        projection_method=projection_method,
        multiscale=multiscale,
    )
    coords = result.astype(np.float32, copy=False)
    return save_result_npz(req.output.result_path, coords, version=VERSION_INFO)


if __name__ == "__main__":
    run_plugin(
        {
            "topometry": run_topometry,
        }
    )
