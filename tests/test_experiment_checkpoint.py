import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from drnb.embed.context import EmbedContext
from drnb.eval.base import EvalResult
from drnb.experiment import Experiment, merge_experiments, read_experiment
from drnb.experiment_common import expected_eval_labels, param_signature, short_col
from drnb.experiment_persistence import LazyResult, experiment_dir


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
    exp.methods = [(("dummy", {"params": {}}), "dummy")]
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
    exp.methods = [(("dummy", {"params": {}}), "dummy")]
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


def test_warn_on_existing_in_post_init(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    caplog.set_level(logging.WARNING, logger="drnb")
    calls: list[str] = []
    _install_dummy_pipeline(monkeypatch, calls)

    exp1 = Experiment(name="exp-warn")
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    exp1.add_dataset("ds1")
    exp1.run()
    assert calls == ["ds1"]

    caplog.clear()
    exp2 = Experiment(name="exp-warn")
    assert any(
        "Experiment directory already exists" in rec.message for rec in caplog.records
    )


def test_clear_storage_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    caplog.set_level(logging.WARNING, logger="drnb")
    calls: list[str] = []
    _install_dummy_pipeline(monkeypatch, calls)

    exp = Experiment(name="exp-clear")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.run()

    caplog.clear()
    exp.clear_storage()
    assert any("Deleting experiment storage" in rec.message for rec in caplog.records)
    assert not (tmp_path / "experiments" / "exp-clear").exists()


def test_status_reports_completed_with_parametrized_evals(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))

    def _create_pipeline(**kwargs):
        class DummyPipeline:
            def __init__(self):
                class Reader:
                    def read_data(self, dataset):
                        return np.zeros((1, 1)), None

                self.reader = Reader()

            def run(self, dataset):
                ctx = EmbedContext(
                    dataset_name=dataset,
                    embed_method_name="dummy",
                    experiment_name="test-exp",
                    drnb_home=tmp_path,
                )
                return {
                    "coords": np.array([[1.0, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="RTE", label="rte-5-euclidean", value=1.0),
                        EvalResult(
                            eval_type="NNP",
                            label="nnp-15-noself-euclidean",
                            value=0.5,
                        ),
                    ],
                    "context": ctx,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline)

    exp = Experiment(name="exp-status")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.evaluations = ["rte", ("nnp", {"n_neighbors": [15]})]

    exp.run()
    status_df = exp.status()
    assert status_df.loc["ds1", "dummy"] == "completed"


def test_expected_eval_labels_expand_parametrized_evals():
    labels = expected_eval_labels(["rte", ("nnp", {"n_neighbors": [15, 50]})])
    assert labels == ["rte-5", "nnp-15", "nnp-50"]


def test_merge_allows_union_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp1 = Experiment(name="exp1")
    exp1.add_dataset("ds1")
    exp1.evaluations = ["rte", ("nnp", {"n_neighbors": [15, 50]})]
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    sig1 = param_signature(exp1.methods[0][0], exp1.evaluations)
    exp1.results = {
        "dummy": {
            "ds1": {
                "coords": np.array([[0.0, 0.0]]),
                "evaluations": [EvalResult(eval_type="dummy", label="m1", value=1.0)],
                "context": None,
            }
        }
    }
    exp1.run_info = {
        "dummy": {"ds1": {"status": "completed", "signature": sig1, "shard": ""}}
    }

    exp2 = Experiment(name="exp2")
    exp2.add_dataset("ds2")
    exp2.evaluations = [
        "rte",
        ("nnp", {"n_neighbors": [15, 50]}),
        ("rpc", {"metric": "euclidean"}),
    ]
    exp2.add_method(("dummy", {"params": {}}), name="dummy")
    sig2 = param_signature(exp2.methods[0][0], exp2.evaluations)
    exp2.results = {
        "dummy": {
            "ds2": {
                "coords": np.array([[0.0, 0.0]]),
                "evaluations": [EvalResult(eval_type="dummy", label="m2", value=2.0)],
                "context": None,
            }
        }
    }
    exp2.run_info = {
        "dummy": {"ds2": {"status": "completed", "signature": sig2, "shard": ""}}
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
    # run_info signatures align with merged evaluations
    method = next(m for m, n in merged.methods if n == "dummy")
    sig = param_signature(method, merged.evaluations)
    assert merged.run_info["dummy"]["ds1"]["signature"] == sig
    assert merged.run_info["dummy"]["ds2"]["signature"] == sig


def test_merge_overwrites_existing_shards(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))

    # Source experiment with real shard
    exp1 = Experiment(name="exp1")
    exp1.drnb_home = tmp_path
    exp1.add_dataset("ds1")
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    exp1.evaluations = ["rte"]
    res1 = {
        "coords": np.array([[1.0, 0.0]]),
        "evaluations": [
            EvalResult(eval_type="RTE", label="rte-5-euclidean", value=1.0)
        ],
        "context": None,
    }
    sig1 = param_signature(exp1.methods[0][0], exp1.evaluations)
    shard_rel = exp1._write_result_shard("dummy", "ds1", res1)
    exp1.results = {"dummy": {"ds1": res1}}
    exp1.run_info = {
        "dummy": {
            "ds1": {"status": "completed", "signature": sig1, "shard": str(shard_rel)}
        }
    }

    # Pre-existing merged directory with stale shard
    stale_res = {
        "coords": np.array([[9.0, 0.0]]),
        "evaluations": [EvalResult(eval_type="stale", label="stale", value=9.0)],
        "context": None,
    }
    stale_exp = Experiment(name="merged", drnb_home=tmp_path)
    stale_exp._write_result_shard("dummy", "ds1", stale_res)

    merged = merge_experiments(
        exp1, Experiment(name="exp2"), name="merged", overwrite=True
    )
    result = merged.results["dummy"]["ds1"]
    if isinstance(result, LazyResult):
        result = result.materialize()
    # The stale shard should have been replaced by the source shard
    assert np.allclose(result["coords"], res1["coords"])


def test_merge_fails_when_dest_exists_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp1 = Experiment(name="exp1")
    exp1.drnb_home = tmp_path
    exp1.add_dataset("ds1")
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    exp1.evaluations = []
    res1 = {"coords": np.array([[0.0, 0.0]]), "context": None}
    shard_rel = exp1._write_result_shard("dummy", "ds1", res1)
    sig1 = param_signature(exp1.methods[0][0], exp1.evaluations)
    exp1.results = {"dummy": {"ds1": res1}}
    exp1.run_info = {
        "dummy": {
            "ds1": {"status": "completed", "signature": sig1, "shard": str(shard_rel)}
        }
    }

    # create target dir to trigger the safety check
    experiment_dir("merged", tmp_path, create=True)

    with pytest.raises(ValueError):
        merge_experiments(exp1, Experiment(name="exp2"), name="merged")


def test_merge_raises_on_method_config_conflict(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp1 = Experiment(name="exp1")
    exp1.add_dataset("ds1")
    exp1.add_method(("dummy", {"params": {}}), name="dup")

    exp2 = Experiment(name="exp2")
    exp2.add_dataset("ds2")
    exp2.add_method(("dummy", {"params": {"alpha": 1}}), name="dup")

    with pytest.raises(ValueError):
        merge_experiments(exp1, exp2, name="merged-conflict", overwrite=True)


def test_partial_eval_rerun_only_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))

    calls = []

    class DummyEvaluator:
        def __init__(self, label):
            self.label = label

        def evaluate(self, X, coords, ctx=None):
            calls.append(self.label)
            return EvalResult(eval_type="dummy", label=self.label, value=1.0)

        def __str__(self):
            return self.label

    def _create_pipeline(**kwargs):
        class DummyPipeline:
            def __init__(self):
                self.evaluators = [DummyEvaluator("ev1"), DummyEvaluator("ev2")]

                class Reader:
                    def read_data(self, dataset):
                        return np.zeros((1, 1)), None

                self.reader = Reader()

            def run(self, dataset):
                return {
                    "coords": np.array([[0.0, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="dummy", label="ev1", value=1.0)
                    ],
                    "context": None,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline)
    monkeypatch.setattr("drnb.experiment.expected_eval_labels", lambda _: ["ev1", "ev2"])
    monkeypatch.setattr("drnb.experiment_common.expected_eval_labels", lambda _: ["ev1", "ev2"])
    monkeypatch.setattr("drnb.experiment_merge.expected_eval_labels", lambda _: ["ev1", "ev2"])

    exp = Experiment(name="exp-partial-evals")
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")

    exp.run()
    assert calls == []  # pipeline provided ev1; no extra evals run

    calls.clear()
    exp.run()
    # Only missing ev2 should be evaluated
    assert calls == ["ev2"]

    status_df = exp.status()
    assert status_df.loc["ds1", "dummy"] == "completed"


def test_partial_eval_with_lazy_result(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []

    class DummyEvaluator:
        def __init__(self, label):
            self.label = label

        def evaluate(self, X, coords, ctx=None):
            calls.append(self.label)
            return EvalResult(eval_type="dummy", label=self.label, value=1.0)

        def __str__(self):
            return self.label

    def _create_pipeline(**kwargs):
        class DummyPipeline:
            def __init__(self):
                self.evaluators = [DummyEvaluator("ev1"), DummyEvaluator("ev2")]

                class Reader:
                    def read_data(self, dataset):
                        return np.zeros((1, 1)), None

                self.reader = Reader()

            def run(self, dataset):
                return {
                    "coords": np.array([[0.0, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="dummy", label="ev1", value=1.0)
                    ],
                    "context": None,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline)
    monkeypatch.setattr("drnb.experiment.expected_eval_labels", lambda _: ["ev1", "ev2"])
    monkeypatch.setattr("drnb.experiment_common.expected_eval_labels", lambda _: ["ev1", "ev2"])
    monkeypatch.setattr("drnb.experiment_merge.expected_eval_labels", lambda _: ["ev1", "ev2"])

    exp1 = Experiment(name="exp1", drnb_home=tmp_path)
    exp1.add_method(("dummy", {"params": {}}), name="dummy")
    exp1.add_dataset("ds1")
    exp1.evaluations = ["ev1", "ev2"]
    res1 = {
        "coords": np.array([[0.0, 0.0]]),
        "evaluations": [EvalResult(eval_type="dummy", label="ev1", value=1.0)],
        "context": None,
    }
    sig1 = param_signature(exp1.methods[0][0], exp1.evaluations)
    shard_rel = exp1._write_result_shard("dummy", "ds1", res1)
    exp1.results = {"dummy": {"ds1": res1}}
    exp1.run_info = {
        "dummy": {
            "ds1": {
                "status": "evals_partial",
                "signature": sig1,
                "shard": str(shard_rel),
            }
        }
    }

    merged = merge_experiments(
        exp1, Experiment(name="exp2"), name="merged", overwrite=True
    )
    calls.clear()
    merged.run()
    assert calls == ["ev2"]  # only missing eval was computed
    result = merged.results["dummy"]["ds1"]
    if isinstance(result, LazyResult):
        result = result.materialize()
    labels = sorted(short_col(ev.label) for ev in result.get("evaluations", []))
    assert labels == ["ev1", "ev2"]


def test_add_evaluations_accepts_single_and_list():
    exp = Experiment(name="exp-add-evals")
    exp.evaluations = ["rte"]
    exp.add_evaluations(("nnp", {"n_neighbors": [15]}))
    exp.add_evaluations(["rte", ("nnp", {"n_neighbors": [15]}), "rpc"])
    assert exp.evaluations == ["rte", ("nnp", {"n_neighbors": [15]}), "rpc"]


def test_shard_dir_defaults_when_shard_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    exp = Experiment(name="exp-shard", drnb_home=tmp_path)
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.run_info = {
        "dummy": {"ds1": {"status": "missing", "signature": "sig", "shard": ""}}
    }
    res = {"coords": np.array([[0.0, 0.0]]), "context": None}
    shard_rel = exp._write_result_shard("dummy", "ds1", res)
    expected_dir = tmp_path / "experiments" / "exp-shard" / shard_rel
    assert expected_dir.is_dir()
    assert "results" in expected_dir.parts


def test_manifest_written_per_dataset_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    calls: list[str] = []

    def _create_pipeline_fail(**kwargs):
        class DummyPipeline:
            def run(self, dataset):
                calls.append(dataset)
                if dataset == "ds2":
                    raise RuntimeError("boom")
                ctx = EmbedContext(
                    dataset_name=dataset,
                    embed_method_name="dummy",
                    experiment_name="exp-crash",
                    drnb_home=tmp_path,
                )
                return {
                    "coords": np.array([[1.0, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="dummy", label="score", value=1.0)
                    ],
                    "context": ctx,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline_fail)

    exp = Experiment(name="exp-crash", drnb_home=tmp_path)
    exp.add_method(("dummy", {"params": {}}), name="dummy")
    exp.add_dataset("ds1")
    exp.add_dataset("ds2")

    with pytest.raises(RuntimeError):
        exp.run()

    manifest_path = tmp_path / "experiments" / "exp-crash" / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "ds1" in manifest["run_info"].get("dummy", {})

    calls.clear()

    def _create_pipeline_ok(**kwargs):
        class DummyPipeline:
            def run(self, dataset):
                calls.append(dataset)
                ctx = EmbedContext(
                    dataset_name=dataset,
                    embed_method_name="dummy",
                    experiment_name="exp-crash",
                    drnb_home=tmp_path,
                )
                return {
                    "coords": np.array([[2.0, 0.0]]),
                    "evaluations": [
                        EvalResult(eval_type="dummy", label="score", value=2.0)
                    ],
                    "context": ctx,
                }

        return DummyPipeline()

    monkeypatch.setattr("drnb.embed.pipeline.create_pipeline", _create_pipeline_ok)

    loaded = read_experiment("exp-crash")
    loaded.run()

    assert calls == ["ds2"]
