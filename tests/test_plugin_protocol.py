from pathlib import Path

import numpy as np
import pytest
from drnb_plugin_sdk import sanitize_params

from drnb.embed.context import EmbedContext
from drnb.plugins.protocol import (
    context_from_payload,
    context_to_payload,
)


def test_sanitize_params_converts_numpy_and_path_types() -> None:
    params = {
        "int": np.int64(3),
        "float": np.float32(1.25),
        "bool": np.bool_(True),
        "path": Path("/tmp/data.npy"),
        "nested": {"values": [np.int32(2), "ok"]},
    }
    sanitized = sanitize_params(params)
    assert sanitized == {
        "int": 3,
        "float": pytest.approx(1.25),
        "bool": True,
        "path": "/tmp/data.npy",
        "nested": {"values": [2, "ok"]},
    }


def test_sanitize_params_rejects_unsupported_types() -> None:
    class Foo:
        pass

    with pytest.raises(TypeError):
        sanitize_params({"bad": Foo()})


def test_context_round_trip() -> None:
    ctx = EmbedContext(
        dataset_name="digits",
        embed_method_name="pacmap-plugin",
        embed_method_variant="debug",
        drnb_home=Path("/tmp/drnb"),
        data_sub_dir="data",
        nn_sub_dir="nn",
        triplet_sub_dir="trip",
        experiment_name="exp1",
    )
    payload = context_to_payload(ctx)
    restored = context_from_payload(payload)
    assert restored is not None
    assert restored.dataset_name == ctx.dataset_name
    assert restored.embed_method_name == ctx.embed_method_name
    assert restored.embed_method_variant == ctx.embed_method_variant
    assert restored.drnb_home == ctx.drnb_home
