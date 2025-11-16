# Plugin Runner Protocol

This document describes the contract between `drnb` (the host) and every external
plugin runner. All plugins must follow these rules so
`src/drnb/plugins/external.py` can launch them safely.

## Request layout

`ExternalEmbedder` writes a JSON request file alongside the serialized inputs in a
temporary workspace. The payload mirrors `drnb_plugin_sdk.protocol.PluginRequest`
and always contains:

- `protocol_version`: currently `1`. Plugins must validate this integer before
  doing any work.
- `method`: the registry key (e.g. `pacmap-plugin`). Use it to dispatch within a
  multi-method runner.
- `params`: the embedding parameters dictionary (JSON primitives only).
- `context`: optional metadata (dataset name, experiment directories). Plugins
  may use this for logging only.
- `input`: paths to serialized arrays:
  - `x_path`: NumPy `.npy` file with the feature matrix.
  - `init_path`: optional `.npy` initialization array.
  - `neighbors`: optional precomputed KNN arrays (`idx_path`, `dist_path`).
- `input.source_paths` (optional): the original on-disk locations for the same
  inputs, when known. Fields include:
  - `drnb_home`: data root.
  - `dataset`, `data_sub_dir`, `nn_sub_dir`, `triplet_sub_dir`: layout hints.
  - `x_path`, `init_path`, and `neighbors.idx_path/dist_path`: canonical files
    under `drnb_home` (e.g., `<drnb_home>/data/<dataset>-data.npy` or
    `<drnb_home>/nn/<name>.<k>.<metric>.<exact|approximate>.<method>.idx.npy`).
- `options`: flags such as `use_precomputed_knn`. More keys may appear over time:
  - `use_sandbox_copies` (default: false) keeps the legacy behavior of copying
    inputs into the plugin workspace. When false (the default), `x_path` and
    neighbor paths point directly at the source files under `DRNB_HOME`.
- `output.result_path`: where the plugin must write its `.npz` result.
- `output.response_path`: where the plugin must write the final JSON response.

## Plugin responsibilities

1. Read the request JSON, either manually or via `drnb_plugin_sdk.helpers.runner`
   (`run_plugin` loads the request for you and passes a `PluginRequest` object to
   your handler).
2. Produce embeddings using **only** the serialized files and parameters in the
   request. Plugins must not import `drnb` core modules.
3. Write the result `.npz` to `req.output.result_path`. Use
   `drnb_plugin_sdk.helpers.results.save_result_npz` to store `coords` and
   optional `snap_*` arrays in a consistent format.
4. Write the final JSON response (e.g., `{"ok": true, "result_npz": ...}` or
   `{"ok": false, "message": ...}`) to `req.output.response_path`. The helper
   `drnb_plugin_sdk.helpers.results.write_response_json` handles this; it's
   called automatically when you use `helpers.runner.run_plugin`.
5. Emit diagnostic logging to stdout and/or stderr freely. The host streams both
   pipes into its log UI, so plugin authors can rely on standard `print` calls
   for progress updates.

## Host expectations

`ExternalEmbedder` launches each plugin using `uv run --quiet --color never` from
the plugin directory, after stripping `VIRTUAL_ENV` so uv selects the plugin's
`.venv`. Stdout and stderr are streamed into the host log with separate prefixes
so users can tell where each line originated. After the process exits, the host
reads `output.response_path` and loads the JSON payload. If the file is missing
or invalid, the run is treated as a fatal error.

The host also validates that the result `.npz` stays inside the workspace and
loads it via `numpy.load` to build the final `EmbedResult`.

## Recommended helpers

- `drnb_plugin_sdk.helpers.runner.run_plugin` – CLI boilerplate, request loading,
  uniform error handling.
- `drnb_plugin_sdk.helpers.logging.log` – emits stderr logs with flushing so
  plugin progress messages appear in the host log immediately.
- `drnb_plugin_sdk.helpers.results.save_result_npz` – writes coords/snapshots as
  float32 and returns the correct response dict.

Using these helpers keeps every plugin aligned with the protocol and avoids
copy/pasted serialization code.
