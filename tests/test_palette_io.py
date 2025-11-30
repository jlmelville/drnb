"""Tests for palette serialization and backward-compatible loading."""

import numpy as np
import pandas as pd

from drnb.io import write_pickle
from drnb.io.dataset import read_palette
from drnb.io.pipeline import DatasetPipeline


def test_process_target_writes_json_palette(monkeypatch, tmp_path):
    """DatasetPipeline should write palettes as JSON and read_palette should load them."""
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    pipeline = DatasetPipeline(
        drnb_home=tmp_path, data_exporters=[], target_exporters=[]
    )
    target = pd.DataFrame({"label": ["a", "b"]})
    dropna_index = np.array([True, True])
    palette = {"label": {"a": "#000000", "b": "#ffffff"}}

    _, target_paths = pipeline.process_target(
        target,
        name="json-palette",
        dropna_index=dropna_index,
        target_cols=["label"],
        target_palette=palette,
    )

    expected_path = tmp_path / "data" / "json-palette-target-palette.json"
    assert expected_path.exists()
    assert target_paths and target_paths[0].endswith("json-palette-target-palette.json")
    loaded = read_palette("json-palette", drnb_home=tmp_path, sub_dir="data")
    assert loaded == palette


def test_read_palette_falls_back_to_pickle(monkeypatch, tmp_path):
    """read_palette should still load legacy pickle palettes if no JSON exists."""
    monkeypatch.setenv("DRNB_HOME", str(tmp_path))
    palette = {"label": {"x": "#123456"}}
    write_pickle(
        palette,
        "pickle-only",
        suffix="target-palette",
        drnb_home=tmp_path,
        sub_dir="data",
        create_sub_dir=True,
    )

    loaded = read_palette("pickle-only", drnb_home=tmp_path, sub_dir="data")
    assert loaded == palette
