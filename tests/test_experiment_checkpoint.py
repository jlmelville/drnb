import json
import os
from pathlib import Path

import numpy as np
import pytest

from drnb.embed.context import EmbedContext
from drnb.eval.base import EvalResult
from drnb.experiment import (
    Experiment,
    LazyResult,
    merge_experiments,
    read_experiment,
)


def _install_dummy_pipeline(monkeypatch, calls):
    def _create_pipeline(**kwargs):
        class DummyPipeline:
            def run(self, dataset):
                idx = len(calls) + 1
                calls.append(dataset)
                ctx = EmbedContext(
                    dataset_name=dataset,
                    embed_method_name="dummy",
                    experiment_name="test-exp",
                    drnb_home=Path(os.environ.get("DRNB_HOME", ".")),
                )
                return {
                    "coords": np.array([[idx, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="dummy", label="score", value=float(idx))
                    ],
                    "context": ctx,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline)


def test_manifest_and_lazy_load(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []
    _install_dummy_pipeline(monkeypatch, calls)

    exp = Experiment(name="exp-basic")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")

    exp.run()
    assert calls == ["ds1"]

    manifest_path = tmp_path / "experiments" / "exp-basic" / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["format_version"] == 2
    shard_rel = manifest["run_info"]["dummy"]["ds1"]["shard"]
    shard_dir = manifest_path.parent / shard_rel
    assert (shard_dir / "result.json").exists()
    assert (shard_dir / "coords.npz").exists()

    loaded = read_experiment("exp-basic")
    assert isinstance(loaded.results["dummy"]["ds1"], LazyResult)

    df = loaded.to_df()
    assert ("dummy", "score") in df.columns
    assert df.loc["ds1", ("dummy", "score")] == pytest.approx(1.0)


def test_rerun_on_mismatch_and_clear(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []
    _install_dummy_pipeline(monkeypatch, calls)

    exp = Experiment(name="exp-resume")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")

    exp.run()
    assert len(calls) == 1

    exp.run()
    assert len(calls) == 1  # skipped because signature matches

    exp.methods = [(("dummy", {"params": {"alpha": 1}}), "dummy")]
    exp.uniq_method_names = {"dummy"}
    exp.run()
    assert len(calls) == 2  # reran due to signature mismatch
    df = exp.to_df()
    assert df.loc["ds1", ("dummy", "score")] == pytest.approx(2.0)

    exp.clear_task("dummy", "ds1")
    exp.run()
    assert len(calls) == 3


def test_context_preserved_on_read(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []
    _install_dummy_pipeline(monkeypatch, calls)

    exp = Experiment(name="exp-context")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.run()

    loaded = read_experiment("exp-context")
    result = loaded.results["dummy"]["ds1"]
    if isinstance(result, LazyResult):
        result = result.materialize()
    assert result["context"].dataset_name == "ds1"
    assert result["context"].embed_method_name == "dummy"


def test_to_df_with_no_evaluations(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []

    def _create_pipeline(**kwargs):
        class DummyPipeline:
            def run(self, dataset):
                calls.append(dataset)
                return {"coords": np.array([[0.0, 0.0]])}

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline)

    exp = Experiment(name="exp-no-evals")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.run()

    df = exp.to_df()
    assert df.empty
    assert calls == ["ds1"]


def test_to_df_respects_dataset_order(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp = Experiment(name="exp-order")
    exp.datasets = ["ds1", "ds2"]
    exp.uniq_datasets = set(exp.datasets)
    exp.methods = [(("dummy", {"params": {}}), "dummy")]
    exp.uniq_method_names = {"dummy"}
    exp.results = {
        "dummy": {
            "ds2": {
                "evaluations": [EvalResult(eval_type="dummy", label="score", value=2.0)]
            },
            "ds1": {
                "evaluations": [EvalResult(eval_type="dummy", label="score", value=1.0)]
            },
        }
    }

    df = exp.to_df()
    assert list(df.index) == ["ds1", "ds2"]


def test_plot_with_partial_results(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    # I accept having to mock away the matplotlib show function because it's not
    # what we are testing for here: we just want to test that an exception is not
    # raised.
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

    exp = Experiment(name="exp-partial")
    exp.datasets = ["ds1", "ds2"]
    exp.uniq_datasets = set(exp.datasets)
    exp.methods = [(("dummy", {"params": {}}), "dummy")]
    exp.uniq_method_names = {"dummy"}
    exp.results = {
        "dummy": {
            "ds1": {
                "coords": np.array([[0.0, 0.0]]),
                "evaluations": [
                    EvalResult(eval_type="dummy", label="score", value=1.0)
                ],
                "context": None,
            }
        }
    }

    fig, _ = plt.subplots()
    exp.plot()
    plt.close(fig)


def test_merge_allows_union_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp1 = Experiment(name="exp1")
    exp1.add_dataset("ds1")
    exp1.evaluations = ["rte", ("nnp", {"n_neighbors": [15, 50]})]
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    exp1.results = {
        "dummy": {
            "ds1": {
                "coords": np.array([[0.0, 0.0]]),
                "evaluations": [EvalResult(eval_type="dummy", label="m1", value=1.0)],
                "context": None,
            }
        }
    }

    exp2 = Experiment(name="exp2")
    exp2.add_dataset("ds2")
    exp2.evaluations = ["rte", ("nnp", {"n_neighbors": [15, 50]}), ("rpc", {"metric": "euclidean"})]
    exp2.add_method(("dummy", {"params": {}}), name="dummy")
    exp2.results = {
        "dummy": {
            "ds2": {
                "coords": np.array([[0.0, 0.0]]),
                "evaluations": [EvalResult(eval_type="dummy", label="m2", value=2.0)],
                "context": None,
            }
        }
    }

    merged = merge_experiments(exp1, exp2, name="merged")
    df = merged.to_df()
    assert list(df.index) == ["ds1", "ds2"]
    assert ("dummy", "m1") in df.columns
    # ds2 has no m1 evaluation so should be NaN, not error
    import pandas as pd

    assert pd.isna(df.loc["ds2", ("dummy", "m1")])
    # evaluations deduped even with unhashable entries
    assert merged.evaluations == [
        "rte",
        ("nnp", {"n_neighbors": [15, 50]}),
        ("rpc", {"metric": "euclidean"}),
    ]
