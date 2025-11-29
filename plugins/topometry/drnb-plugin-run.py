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

VERSION_INFO = build_version_payload(
    package="topometry", plugin_package="drnb-plugin-topometry"
)


def run_topometry(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    log(f"Running topometry with params={summarize_params(params)}")

    if "n_eigs" not in params:
        # the minimum of the shape of x
        n_eigs = min(list(x.shape) + [100])
        log(f"Setting n_eigs to {n_eigs}")
        params["n_eigs"] = n_eigs
    if "cache" not in params:
        params["cache"] = False

    embedder = tp.TopOGraph(**params)
    embedder.fit_transform(x)
    result = embedder.project(projection_method="MAP")
    coords = result.astype(np.float32, copy=False)
    return save_result_npz(req.output.result_path, coords, version=VERSION_INFO)


if __name__ == "__main__":
    run_plugin(
        {
            "topometry": run_topometry,
        }
    )
