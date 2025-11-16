#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import numpy as np
import umato
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.paths import resolve_x_path
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin

DEFAULT_HUB_NUM = 300


def _adjust_hub_num(x: np.ndarray, params: dict[str, Any]) -> None:
    n_samples = x.shape[0]
    hub_num = int(params.get("hub_num", DEFAULT_HUB_NUM))
    if n_samples < DEFAULT_HUB_NUM and hub_num > n_samples:
        adjusted = max(1, n_samples // 3)
        log(
            f"Reducing hub_num from {hub_num} to {adjusted} for dataset size {n_samples}"
        )
        params["hub_num"] = adjusted


def run_umato(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    sdk_protocol.context_from_payload(req.context)

    x = np.load(resolve_x_path(req), allow_pickle=False)
    params = dict(req.params or {})

    _adjust_hub_num(x, params)

    log(f"Running UMATO with params={summarize_params(params)}")
    embedder = umato.UMATO(**params)
    coords = embedder.fit_transform(x).astype(np.float32, copy=False)

    log(f"Saving results to {req.output.result_path}")
    result = save_result_npz(req.output.result_path, coords)
    return result


if __name__ == "__main__":
    run_plugin({"umato-plugin": run_umato})
