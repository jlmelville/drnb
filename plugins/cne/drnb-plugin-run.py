#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import cne
import numpy as np
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.paths import (
    resolve_init_path,
    resolve_neighbors,
    resolve_x_path,
)
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin


def run_cne(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    # init_path = resolve_init_path(req)
    # init = np.load(init_path, allow_pickle=False) if init_path else None
    # neighbors = resolve_neighbors(req)

    log(f"Running cne with params={summarize_params(params)}")

    embedder = cne.CNE(**params)
    result = embedder.fit_transform(x)
    coords = result.astype(np.float32, copy=False)
    return save_result_npz(req.output.result_path, coords)


if __name__ == "__main__":
    run_plugin(
        {
            "cne": run_cne,
        }
    )
