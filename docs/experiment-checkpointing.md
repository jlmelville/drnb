# Experiment Checkpointing (Format v2)

This document describes how experiments are persisted (pickle-free), what files get written, and how checkpointing behaves during runs, merges, and partial evaluation reruns.

## Storage Layout

All files live under `DRNB_HOME/experiments/<name>/`:
- `manifest.json` – top-level metadata and per-(method,dataset) run info.
- `results/<method>/<dataset>/result.json` – shard metadata listing stored fields and how to load them.
- `results/<method>/<dataset>/*.npz` – arrays (e.g., `coords.npz`) stored alongside `result.json`.

## Manifest (v2)

`manifest.json` contains:
- `format_version`: `2`
- `name`, `datasets`, `methods` (serialized configs), `evaluations`
- `run_info`: map of `method -> dataset -> {status, signature, updated_at, shard, evals_completed, evals_expected}`
  - `status`: `missing`, `evals_partial`, or `completed`
  - `signature`: stable hash of method config + evaluations (drives skip/rerun)
  - `shard`: relative path to the shard directory
  - `evals_completed` / `evals_expected`: counts for partial-eval tracking

## Shards

Each shard directory holds:
- `result.json` describing each entry (`coords`, `evaluations`, `context`, etc.) and how to load it (`npz`, `json`, `eval_results`, `context`, etc.).
- One `.npz` per array field (e.g., `coords.npz`).

## Runtime Behavior

- `Experiment.run()` processes each `(method, dataset)`:
  - If no coords: run pipeline (coords + all evals), write shard, update manifest.
  - If signature mismatches: rerun full pipeline and overwrite shard.
  - If coords present, signature matches, and some evals are missing: reuse coords and run only the missing evals; merge into shard.
  - If signature matches and all evals present: skip.
- Status values:
  - `missing`: no coords stored.
  - `evals_partial`: coords present, some evals missing (counts recorded).
  - `completed`: coords present and all expected evals found (0/0 evals also counts as complete).
- `status()` reports a DataFrame; partial evals render as `evals_partial(x/y)`.
- `run_info` is rewritten to `manifest.json` after each dataset; shards are written immediately when produced.

## Merge Behavior

- `merge_experiments(exp1, exp2, name, overwrite=False)`:
  - Copies shards into the destination experiment directory (overwriting shard paths when `overwrite=True`).
  - Recomputes status/eval counts against the merged evaluation set; partial evals stay partial until rerun.
  - Errors if the destination experiment directory already exists and `overwrite` is `False`.
  - Raises a `ValueError` when the same method name exists in both sources with different configurations; rename the method or align configs before merging.

## Helpers

- `add_datasets`, `add_evaluations` (accepts single or list, deduped).
- `clear_task(method, dataset)`: delete one shard/run_info entry so it reruns.
- `clear_method(method)`: delete all shards for a method.
- `clear_storage()`: delete the experiment directory on disk, keep in-memory setup.
- `reset()`: delete storage and clear all in-memory experiment config (destructive).

## Sample Files

`manifest.json` (abridged):
```json
{
  "format_version": 2,
  "name": "demo-exp",
  "datasets": ["iris"],
  "methods": [
    {"name": "pca", "method": {"kind": "json", "value": "pca"}}
  ],
  "evaluations": ["rte", ["nnp", {"n_neighbors": [15]}]],
  "run_info": {
    "pca": {
      "iris": {
        "status": "completed",
        "signature": "8e7c1f...",
        "updated_at": "2025-11-28T12:00:00Z",
        "shard": "results/pca/iris",
        "evals_completed": 2,
        "evals_expected": 2
      }
    }
  }
}
```

`results/pca/iris/result.json` (abridged):
```json
{
  "version": 1,
  "entries": {
    "coords": {"type": "npz", "file": "coords.npz"},
    "evaluations": {
      "type": "eval_results",
      "value": [
        {"eval_type": "RTE", "label": "rte-5-euclidean", "value": 0.9, "info": {}},
        {"eval_type": "NNP", "label": "nnp-15-noself-euclidean", "value": 0.8, "info": {}}
      ]
    },
    "context": {
      "type": "context",
      "value": {"dataset_name": "iris", "embed_method_name": "pca", "experiment_name": "demo-exp"}
    }
  }
}
```

