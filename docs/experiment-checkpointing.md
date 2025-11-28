# Experiment Checkpointing (Format v2)

This document describes how experiments are persisted without pickles, what files get written, and how checkpointing behaves during runs.

## What gets written

- `experiments/<name>/manifest.json` – JSON manifest (format_version=2) that lists datasets, methods, evaluations, and `run_info` for each `(method, dataset)` pair. It is rewritten after every completed dataset.
- `experiments/<name>/results/<method>/<dataset>/result.json` – metadata for a single shard. Lists each stored field and how to load it.
- `experiments/<name>/results/<method>/<dataset>/*.npz` – arrays (e.g., `coords.npz`) stored alongside `result.json`. Only array fields get `.npz` files.

All paths are rooted under `DRNB_HOME/experiments/`.

## Runtime behavior

- When `Experiment.run()` executes a `(method, dataset)` pair, it writes a shard immediately, updates `run_info`, and rewrites `manifest.json`.
- If a shard already exists and its signature matches the current method + evaluations, the pair is skipped.
- If a shard exists but the signature differs, the pair is rerun and the shard is overwritten to keep results aligned with current params.
- `Experiment.results[...]` entries are lazy: they load shard contents only when accessed (e.g., `to_df` or `plot`).

Signatures are stable hashes built from the serialized method configuration (including chained embedders) and the evaluation list. They change when method params or evaluation configuration changes.

## Helper methods

- `clear_task(method_name, dataset)` – deletes one shard and its `run_info` so that pair reruns on the next `run()`.
- `clear_method(method_name)` – deletes all shards for a method so every dataset reruns.
- `reset()` – deletes the entire experiment directory and clears in-memory state (destructive).

## Logging

- When a shard is written: `Checkpointed <method>/<dataset> -> <path>`.
- When the manifest is written: `Wrote manifest for experiment <name> to <path>`.
- Skips and reruns are logged inside `run()` (skip with matching signature; rerun on mismatch).
