import json
import sys
from pathlib import Path

import pytest

from drnb_plugin_sdk.helpers import runner
from drnb_plugin_sdk.protocol import (
    PROTOCOL_VERSION,
    PluginInputPaths,
    PluginOutputPaths,
    PluginRequest,
    PluginOptions,
    PluginNeighbors,
    request_to_dict,
)


def test_run_plugin_exits_on_request_parse_error(monkeypatch, capsys) -> None:
    calls: list[object] = []

    def fake_handler(request: PluginRequest) -> dict[str, object]:
        calls.append(request)
        return {"ok": True}

    def fake_load_request(path: str | Path) -> PluginRequest:
        raise RuntimeError("cannot decode request")

    monkeypatch.setattr(runner, "load_request", fake_load_request)
    monkeypatch.setattr(
        sys, "argv", ["prog", "--method", "demo", "--request", "request.json"]
    )

    with pytest.raises(SystemExit) as excinfo:
        runner.run_plugin({"demo": fake_handler})

    assert excinfo.value.code == 1
    assert calls == []
    err = capsys.readouterr().err
    assert "cannot decode request" in err


def test_run_plugin_writes_error_response_on_handler_exception(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    response_path = tmp_path / "response.json"
    req = PluginRequest(
        protocol_version=PROTOCOL_VERSION,
        method="demo",
        params={"example": 1},
        context=None,
        input=PluginInputPaths(
            x_path=str(tmp_path / "x.npy"),
            neighbors=PluginNeighbors(),
        ),
        options=PluginOptions(),
        output=PluginOutputPaths(
            result_path=str(tmp_path / "result.npz"),
            response_path=str(response_path),
        ),
    )
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request_to_dict(req)), encoding="utf-8")

    def failing_handler(request: PluginRequest) -> dict[str, object]:
        raise ValueError("handler failed")

    monkeypatch.setattr(
        sys, "argv", ["prog", "--method", "demo", "--request", str(req_path)]
    )

    runner.run_plugin({"demo": failing_handler})

    payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert payload == {"ok": False, "message": "handler failed"}
    err = capsys.readouterr().err
    assert "handler failed" in err
