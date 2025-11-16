#!/usr/bin/env python
from __future__ import annotations

import ncvis
import numpy as np
from drnb_plugin_sdk_310 import protocol as sdk_protocol
from drnb_plugin_sdk_310.helpers.logging import log, summarize_params
from drnb_plugin_sdk_310.helpers.paths import resolve_x_path
from drnb_plugin_sdk_310.helpers.results import save_result_npz
from drnb_plugin_sdk_310.helpers.runner import run_plugin


def run_ncvis(req: sdk_protocol.PluginRequest) -> dict[str, str]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    log(f"Running NCVis with params={summarize_params(params)}")
    embedder = ncvis.NCVis(**params)
    coords = embedder.fit_transform(x).astype(np.float32, copy=False)

    return save_result_npz(req.output.result_path, coords)


if __name__ == "__main__":
    run_plugin({"ncvis-plugin": run_ncvis})
