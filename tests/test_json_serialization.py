"""Tests for JSON serialization of dataclasses, verifying behavior before and after removing Jsonizable mixin."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from drnb.eval.globalscore import GlobalScore
from drnb.io import write_json
from drnb.io.pipeline import DatasetPipeline, DatasetPipelineResult
from drnb.neighbors.compute import NeighborsRequest
from drnb.triplets import TripletsRequest


def test_neighbors_request_json_serialization():
    """Test that NeighborsRequest can be serialized to JSON."""
    req = NeighborsRequest(n_neighbors=[10, 20], method="exact", metric="euclidean")
    # Serialize using asdict (what we'll use after removing Jsonizable)
    data = asdict(req)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    # Verify it's valid JSON and contains expected fields
    parsed = json.loads(json_str)
    assert parsed["n_neighbors"] == [10, 20]
    assert parsed["method"] == "exact"
    assert is_dataclass(req)


def test_triplets_request_json_serialization():
    """Test that TripletsRequest can be serialized to JSON."""
    req = TripletsRequest(n_triplets_per_point=5, seed=42)
    data = asdict(req)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed["n_triplets_per_point"] == 5
    assert parsed["seed"] == 42


def test_global_score_json_serialization():
    """Test that GlobalScore can be serialized to JSON."""
    score = GlobalScore()
    data = asdict(score)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)
    # GlobalScore has no fields, but should serialize to empty dict
    assert isinstance(parsed, dict)


def test_dataset_pipeline_json_serialization():
    """Test that DatasetPipeline can be serialized to JSON."""
    pipeline = DatasetPipeline(verbose=True, check_for_duplicates=True)
    data = asdict(pipeline)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed["verbose"] is True
    assert parsed["check_for_duplicates"] is True


def test_dataset_pipeline_result_nested_serialization():
    """Test that DatasetPipelineResult with nested DatasetPipeline serializes correctly."""
    pipeline = DatasetPipeline(verbose=True)
    result = DatasetPipelineResult(pipeline=pipeline, started_on="2024-01-01")
    data = asdict(result)
    # Verify nested dataclass is converted to dict
    assert isinstance(data["pipeline"], dict)
    assert data["pipeline"]["verbose"] is True
    assert data["started_on"] == "2024-01-01"
    # Verify it can be serialized to JSON
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert "pipeline" in parsed
    assert parsed["pipeline"]["verbose"] is True


def test_write_json_with_dataclass(tmp_path: Path):
    """Test that write_json() works with dataclass instances using tmp_path."""
    pipeline = DatasetPipeline(verbose=True)
    # Use tmp_path (pytest fixture that auto-cleans)
    output_path = write_json(pipeline, "test-pipeline", drnb_home=tmp_path)
    # Verify file was created
    assert output_path.exists()
    # Verify it's valid JSON
    with open(output_path) as f:
        data = json.load(f)
    assert data["verbose"] is True
    # Clean up is automatic via tmp_path fixture


def test_write_json_with_dict(tmp_path: Path):
    """Test that write_json() still works with plain dictionaries."""
    data = {"key": "value", "number": 42}
    output_path = write_json(data, "test-dict", drnb_home=tmp_path)
    assert output_path.exists()
    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_json_nested_dataclass(tmp_path: Path):
    """Test write_json() with nested dataclass (DatasetPipelineResult)."""
    pipeline = DatasetPipeline(verbose=True)
    result = DatasetPipelineResult(pipeline=pipeline, started_on="2024-01-01")
    output_path = write_json(result, "test-result", drnb_home=tmp_path)
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert "pipeline" in data
    assert isinstance(data["pipeline"], dict)
    assert data["pipeline"]["verbose"] is True
    assert data["started_on"] == "2024-01-01"
