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
        "tuple": (np.int16(1), "b"),
        "set": {np.float64(2.5), "a"},
    }
    sanitized = sanitize_params(params)
    assert sanitized["int"] == 4
    assert sanitized["float"] == pytest.approx(1.5)
    assert sanitized["string"] == "ok"
    assert sanitized["path"].endswith("file.npy")
    assert sanitized["nested"]["values"][0] == 2
    assert sanitized["tuple"] == [1, "b"]
    assert "a" in sanitized["set"]
    assert any(val == pytest.approx(2.5) for val in sanitized["set"])


def test_sanitize_params_rejects_unsupported_types() -> None:
    class Foo:
        pass

    with pytest.raises(TypeError, match="params.bad"):
        sanitize_params({"bad": Foo()})


def test_request_round_trip(tmp_path: Path) -> None:
    ctx = PluginContext(
        dataset_name="toy", embed_method_name="pacmap", drnb_home=tmp_path
    )
    payload = context_to_payload(ctx)
    restored = context_from_payload(payload)
    assert restored == ctx

    req = PluginRequest(
        protocol_version=PROTOCOL_VERSION,
        method="pacmap",
        params={"n_neighbors": 10},
        context=payload,
        input=PluginInputPaths(
            x_path=str(tmp_path / "x.npy"),
            neighbors=PluginNeighbors(idx_path="idx.npy", dist_path="dist.npy"),
        ),
        options=PluginOptions(use_sandbox_copies=True),
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
    assert loaded.options.use_sandbox_copies is True


def test_request_without_optional_paths(tmp_path: Path) -> None:
    req = PluginRequest(
        protocol_version=PROTOCOL_VERSION,
        method="tsne",
        params={"perplexity": 30},
        context=None,
        input=PluginInputPaths(
            x_path=str(tmp_path / "x.npy"),
            neighbors=PluginNeighbors(idx_path=None, dist_path=None),
        ),
        options=PluginOptions(use_sandbox_copies=False),
        output=PluginOutputPaths(
            result_path="result.npz", response_path="response.json"
        ),
    )
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request_to_dict(req)), encoding="utf-8")

    loaded = load_request(req_path)
    assert loaded.method == "tsne"
    assert loaded.input.neighbors.idx_path is None
