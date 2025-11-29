# Plugin Runner Protocol

This document describes the contract between `drnb` (the host) and every external
plugin runner. All plugins must follow these rules so
`src/drnb/plugins/external.py` can launch them safely.

## Location

The embedder plugins all live under the `plugins` folder. The SDK that supports the
protocol for sharing data between a plugin and the drnb core is in `plugin-sdks`. There is one
SDK per python version, so as and when I decide to migrate to newer versions of python new SDKs
will appear with the python version at the end e.g. `drnb-plugin-sdk-312` is for Python 3.12.
This allows both the drnb core and the plugins to vary the version of python they support.

## Request layout

`ExternalEmbedder` writes a JSON request file alongside the serialized inputs in a
temporary workspace. The payload mirrors `drnb_plugin_sdk.protocol.PluginRequest`
and always contains:

- `protocol_version`: currently `1`. Plugins must validate this integer before doing any work.
- `method`: the registry key (e.g. `pacmap`, `tsne`). No `-plugin` suffix.
- `params`: the embedding parameters dictionary (JSON primitives only).
- `context`: metadata for logging; typically `dataset_name`, `embed_method_name` (matches `method`), optional `embed_method_variant`, `experiment_name`, and layout hints (`drnb_home`, `data_sub_dir`, `nn_sub_dir`, `triplet_sub_dir`).
- `input`: authoritative paths plugins must read:
  - `x_path`: required feature matrix path.
  - `init_path`: optional initialization path (may be `null`).
  - `neighbors`: optional precomputed KNN (`idx_path`, `dist_path`, either may be `null`).
- `options`: flags such as `use_precomputed_knn`. Current keys:
  - `use_sandbox_copies` (default: false) copies inputs into the workspace; `input.*` points at those copies.
- `keep_temps`, `use_precomputed_knn`, `use_sandbox_copies` remain available.
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
   `{"ok": false, "message": ...}`) to `req.output.response_path`. Include a
   `version` payload describing the embedder library and, optionally, the plugin
   runner itself. A typical payload looks like
   `{"package": "pacmap", "version": "0.8.2", "plugin_package": "drnb-plugin-pacmap", "plugin_version": "0.0.1"}`.
   Use the SDK helper `helpers.version.build_version_payload` (which relies on
   `importlib.metadata.version`) to avoid parsing lock files or pyproject tables.
   The helper `save_result_npz` accepts this payload via its `version` argument.
   The helper
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
- `drnb_plugin_sdk.helpers.logging.log` – emits stdout logs with flushing so
  plugin progress messages appear in the host log immediately.
- `drnb_plugin_sdk.helpers.results.save_result_npz` – writes coords/snapshots as
  float32 and returns the correct response dict.

Using these helpers keeps every plugin aligned with the protocol and avoids
copy/pasted serialization code.

### Sample request (zero-copy)

  {
    "protocol": 1,
    "method": "tsne",
    "params": {
      "dof": 0.7,
      "initialization": "spectral"
    },
    "context": {
      "dataset_name": "s1k",
      "embed_method_name": "tsne",
      "embed_method_variant": "",
      "drnb_home": "/home/data/datasets",
      "data_sub_dir": "data",
      "nn_sub_dir": "nn",
      "triplet_sub_dir": "triplets",
      "experiment_name": "pipeline-20251116234807"
    },
    "input": {
      "x_path": "/home/data/datasets/data/s1k-data.npy",
      "init_path": null,
      "neighbors": {
        "idx_path": "/home/data/datasets/nn/s1k.151.euclidean.exact.faiss.idx.npy",
        "dist_path": "/home/data/datasets/nn/s1k.151.euclidean.exact.faiss.dist.npy"
      }
    },
    "options": {
      "keep_temps": false,
      "use_precomputed_knn": true,
      "use_sandbox_copies": false
    },
    "output": {
      "result_path": "/tmp/drnb-tsne-XXXX/result.npz",
      "response_path": "/tmp/drnb-tsne-XXXX/response.json"
    }
  }

### Sample response with version metadata

  {
    "ok": true,
    "result_npz": "/tmp/drnb-tsne-XXXX/result.npz",
    "version": {
      "package": "openTSNE",
      "version": "1.0.0",
      "plugin_package": "drnb-plugin-tsne",
      "plugin_version": "0.0.1"
    }
  }

## Exploring plugins via notebooks

Directly using plugin packages in the main drnb core is no longer possible, so if you like playing
with them directly, you will want to:

- activate the plugin virtual environment `source .venv/bin/activate` in the plugin folder
- `uv add ipykernel`
- register the env as kernel e.g. for PyMDE: `python -m ipykernel install --user --name drnb-pymde --display-name "drnb: PyMDE"`

You should then be able to pick the kernel from however you interact with notebooks (this should
work with e.g. VSCode notebook support too).
