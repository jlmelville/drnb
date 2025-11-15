import json
from pathlib import Path

import numpy as np
import pytest
from drnb_plugin_sdk.protocol import (
    PROTOCOL_VERSION,
    PluginContext,
    PluginInputPaths,
    PluginNeighbors,
    PluginOptions,
    PluginOutputPaths,
    PluginRequest,
    context_from_payload,
    context_to_payload,
    load_request,
    request_to_dict,
    sanitize_params,
)


def test_sanitize_params_handles_numpy_and_paths(tmp_path: Path) -> None:
    params = {
        "int": np.int64(4),
        "float": np.float32(1.5),
        "string": "ok",
        "path": tmp_path / "file.npy",
        "nested": {"values": [np.int32(2), "x"]},
    }
    sanitized = sanitize_params(params)
    assert sanitized["int"] == 4
    assert sanitized["float"] == pytest.approx(1.5)
    assert sanitized["path"].endswith("file.npy")
    assert sanitized["nested"]["values"][0] == 2


def test_request_round_trip(tmp_path: Path) -> None:
    ctx = PluginContext(dataset_name="toy", embed_method_name="pacmap")
    payload = context_to_payload(ctx)
    restored = context_from_payload(payload)
    assert restored == ctx

    req = PluginRequest(
        protocol_version=PROTOCOL_VERSION,
        method="pacmap-plugin",
        params={"n_neighbors": 10},
        context=payload,
        input=PluginInputPaths(
            x_path=str(tmp_path / "x.npy"),
            neighbors=PluginNeighbors(idx_path="idx.npy", dist_path="dist.npy"),
        ),
        options=PluginOptions(),
        output=PluginOutputPaths(
            result_path="result.npz", response_path="response.json"
        ),
    )
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request_to_dict(req)), encoding="utf-8")

    loaded = load_request(req_path)
    assert loaded.method == req.method
    assert loaded.params == req.params
    assert loaded.input.x_path == req.input.x_path
    assert loaded.output.result_path == "result.npz"
    assert loaded.output.response_path == "response.json"
