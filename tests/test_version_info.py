from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import numpy as np

from drnb.embed.pca import Pca
from drnb.embed.version import UNKNOWN_VALUE, get_embedder_version_info
from drnb.experiment import Experiment


def test_get_embedder_version_info_prefers_metadata(monkeypatch) -> None:
    calls: list[str] = []

    def fake_version(name: str) -> str:
        calls.append(name)
        mapping = {"scikit-learn": "1.2.3"}
        if name in mapping:
            return mapping[name]
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        "drnb.embed.version.metadata",
        SimpleNamespace(
            version=fake_version,
            PackageNotFoundError=importlib_metadata.PackageNotFoundError,
        ),
    )

    info = get_embedder_version_info(Pca(), "pca")
    assert info["package"] == "scikit-learn"
    assert info["version"] == "1.2.3"
    assert "scikit-learn" in calls


def test_experiment_versions_returns_dict_and_df() -> None:
    exp = Experiment(name="versions-exp")
    exp.datasets = ["ds1", "ds2"]
    exp.methods = [(("dummy", {"params": {}}), "dummy")]

    exp.run_info = {
        "dummy": {
            "ds1": {
                "version_info": {"package": "pkg1", "version": "0.1", "source": "core"}
            }
        }
    }
    exp.results = {
        "dummy": {
            "ds1": {"coords": np.zeros((1, 2), dtype=np.float32)},
            "ds2": {
                "coords": np.zeros((1, 2), dtype=np.float32),
                "version_info": {
                    "package": "pkg2",
                    "version": "2.0",
                },
            },
        }
    }

    versions = exp.versions()
    assert versions["dummy"]["ds1"]["package"] == "pkg1"
    assert versions["dummy"]["ds1"]["version"] == "0.1"
    assert versions["dummy"]["ds2"]["version"] == "2.0"

    df = exp.versions(as_df=True)
    assert set(df["method"]) == {"dummy"}
    assert set(df["dataset"]) == {"ds1", "ds2"}
    row_ds2 = df[df["dataset"] == "ds2"].iloc[0]
    assert row_ds2["package"] == "pkg2"
    assert row_ds2["version"] == "2.0"

    unknown_exp = Experiment(name="empty")
    assert unknown_exp.versions() == {}
    empty_df = unknown_exp.versions(as_df=True)
    assert empty_df.empty
    assert list(empty_df.columns) == [
        "method",
        "dataset",
        "package",
        "version",
        "component",
    ]
