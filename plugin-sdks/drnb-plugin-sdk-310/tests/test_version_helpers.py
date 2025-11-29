from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import numpy as np

from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.version import (
    UNKNOWN_VALUE,
    build_version_payload,
    get_package_version,
)


def test_build_version_payload_prefers_metadata(monkeypatch) -> None:
    calls: list[str] = []

    def fake_version(name: str) -> str:
        calls.append(name)
        mapping = {"demo": "1.2.3", "plugin-demo": "9.9.9"}
        if name in mapping:
            return mapping[name]
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        "drnb_plugin_sdk.helpers.version.metadata",
        SimpleNamespace(
            version=fake_version, PackageNotFoundError=importlib_metadata.PackageNotFoundError
        ),
    )

    payload = build_version_payload("demo", plugin_package="plugin-demo")
    assert payload == {
        "package": "demo",
        "version": "1.2.3",
        "plugin_package": "plugin-demo",
        "plugin_version": "9.9.9",
    }
    assert "demo" in calls
    assert "plugin-demo" in calls


def test_get_package_version_returns_none_on_missing(monkeypatch) -> None:
    def missing(name: str):
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        "drnb_plugin_sdk.helpers.version.metadata",
        SimpleNamespace(
            version=missing, PackageNotFoundError=importlib_metadata.PackageNotFoundError
        ),
    )
    assert get_package_version("absent") is None
    payload = build_version_payload("absent")
    assert payload["version"] == UNKNOWN_VALUE


def test_save_result_npz_carries_version(tmp_path) -> None:
    coords = np.zeros((2, 2), dtype=np.float32)
    resp = save_result_npz(
        tmp_path / "coords.npz", coords, version={"package": "demo", "version": "0.1"}
    )
    assert resp["ok"] is True
    assert resp["version"]["package"] == "demo"
    assert resp["version"]["version"] == "0.1"
