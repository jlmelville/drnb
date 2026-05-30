# drnb Maintenance

This document records the repository maintenance contract: workspace layout, Python policy,
lockfile usage, dependency upgrade order, and CI scope.

## Workspace Matrix

Core and SDK workspaces:

| Workspace | Python requirement | Lockfile | SDK dependency | Install and CI intent |
| --- | --- | --- | --- | --- |
| `.` | `>=3.12,<3.13` | `uv.lock` | `drnb-plugin-sdk-312`, `drnb-nn-plugin-sdk-312` | Strict install and full root checks. |
| `plugin-sdks/drnb-plugin-sdk-312` | `>=3.12,<3.13` | `plugin-sdks/drnb-plugin-sdk-312/uv.lock` | None | Strict install, SDK tests, lock check. |
| `plugin-sdks/drnb-plugin-sdk-310` | `==3.10.14` | `plugin-sdks/drnb-plugin-sdk-310/uv.lock` | None | Legacy SDK for `ncvis`; lock check and targeted tests. |
| `nn-plugin-sdks/drnb-nn-plugin-sdk-312` | `>=3.12,<3.13` | `nn-plugin-sdks/drnb-nn-plugin-sdk-312/uv.lock` | None | Strict install and lock check; add targeted tests when the NN runner contract changes. |

Embedder plugin workspaces:

| Workspace | Python requirement | SDK dependency | Lockfile | CI intent |
| --- | --- | --- | --- | --- |
| `plugins/pacmap` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/pacmap/uv.lock` | Lock check and lightweight smoke install. |
| `plugins/cne` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/cne/uv.lock` | Lock check; GPU and PyKeOps behavior is manual. |
| `plugins/ncvis` | `==3.10.14` | `drnb-plugin-sdk-310` | `plugins/ncvis/uv.lock` | Legacy/manual; keep Python 3.10 and NumPy <2 until proven otherwise. |
| `plugins/pymde` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/pymde/uv.lock` | Lock check; Torch/GPU behavior is manual. |
| `plugins/topometry` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/topometry/uv.lock` | Lock check and smoke install if dependency resolution remains stable. |
| `plugins/trimap` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/trimap/uv.lock` | Lock check and lightweight smoke install. |
| `plugins/tsne` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/tsne/uv.lock` | Lock check and lightweight smoke install. |
| `plugins/umato` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/umato/uv.lock` | Lock check and lightweight smoke install. |

Nearest-neighbor plugin workspaces:

| Workspace | Python requirement | SDK dependency | Lockfile | CI intent |
| --- | --- | --- | --- | --- |
| `nn-plugins/annoy` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/annoy/uv.lock` | Lock check and lightweight smoke install. |
| `nn-plugins/faiss` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/faiss/uv.lock` | Lock check only; FAISS itself is installed manually. |
| `nn-plugins/hnsw` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/hnsw/uv.lock` | Lock check and lightweight smoke install. |
| `nn-plugins/torchknn` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/torchknn/uv.lock` | Lock check; GPU and PyTorch build behavior is manual. |

## Python Policy

The root package currently targets Python 3.12. A future Python 3.13 migration should be treated as
a compatibility spike first: create 3.13 SDKs, prove lockfile resolution, and migrate plugins
selectively. Do not move `plugins/ncvis` or `plugin-sdks/drnb-plugin-sdk-310` away from Python
3.10 during routine cleanup.

The root `.python-version` is the default for Python 3.12 workspaces. Workspaces that require a
different interpreter should carry their own `.python-version`; `ncvis` is the important legacy
candidate.

## Installation Policy

The core install path is:

```bash
uv sync --locked
```

The full install path is:

```bash
./scripts/install.sh
```

`scripts/install.sh` syncs each selected workspace. Its contract is:

- SDK and root package installs are strict and should fail loudly.
- Syncs use `uv sync --locked` by default.
- `--refresh-locks` runs `uv lock` in each selected workspace before syncing from the refreshed
  lockfile.
- Embedder and NN plugin installs are best effort and are summarized at the end of the run.
- `--reinstall-all` reinstalls all plugins.
- `--reinstall <name>` reinstalls a specific embedder or NN plugin.
- `--fresh` removes each workspace `.venv` before syncing.

## Lockfile Policy

Checked-in `uv.lock` files are the source of truth for normal installs. Use these commands for
routine validation:

```bash
uv lock --check
uv sync --locked
```

Run the same commands inside each SDK or plugin workspace when that workspace is part of the
change.

Use mutating lockfile commands only as maintenance operations:

```bash
uv lock
uv lock --upgrade
./scripts/install.sh --refresh-locks
```

Keep lock refreshes scoped to one workspace or one dependency bucket. Do not combine Python
migration, NumPy migration, Torch migration, and plugin package upgrades in a single blind lockfile
change.

## Dependency Upgrade Buckets

Process dependency changes in this order:

1. Developer tooling and low-risk maintenance packages, such as `pytest` and `ruff`.
2. Core non-GPU runtime packages, such as `glasbey`, `pynndescent`, `umap-learn`, `plotly`, and
   `rich`.
3. Numeric and data stack packages, such as NumPy, numba, llvmlite, SciPy, pandas, pyarrow, and
   scikit-learn.
4. Algorithm plugin packages, such as `pacmap`, `pymde`, and `topometry`.
5. Torch packages, handled separately because CPU, CUDA, MPS, and old-GPU support can diverge.
6. Deferred or rejected upgrades, especially changes that rewrite `ncvis` or the Python 3.10 SDK.

After each bucket, run the narrowest checks that prove the affected behavior, then broaden to the
root checks when the bucket touches shared runtime behavior.

## Renovate Policy

Treat Renovate PRs as useful input, not automatic merge candidates. Before merging:

- Check whether the PR changes Python policy, SDK isolation, or legacy plugin constraints.
- Split broad lockfile updates into the dependency buckets above.
- Keep `ncvis` on Python 3.10 and NumPy <2 until a focused build/run spike proves the migration.
- Treat Torch upgrades as a separate decision with explicit CPU, CUDA, MPS, and old-GPU notes.

Known spring-cleaning PRs:

- `#2` NumPy: mine for compatible updates, but do not merge wholesale while it loosens `ncvis` and
  Python 3.10 SDK constraints.
- `#6` Python Docker tag: reject or supersede because it jumps to Python 3.14 and rewrites legacy
  Python 3.10 workspaces.
- `#8` Torch: evaluate separately after deciding whether old-GPU support still matters.

## CI Scope

There are currently no checked-in GitHub Actions workflows. Initial CI should cover what the repo
can guarantee on hosted runners:

- Root `uv sync --locked`, `uv run ruff check .`, and `uv run pytest`.
- Lock checks for the root, SDKs, and selected plugin workspaces.
- Lightweight plugin smoke installs where dependencies are reliable.
- `actionlint .github/workflows/*.yml` and `uvx zizmor .github/workflows` once workflows exist.

Initial CI should explicitly exclude or document:

- Manual FAISS GPU installation.
- Torch GPU validation.
- Full `ncvis` native build/run checks if hosted runners prove slow or brittle.

Pin third-party GitHub Actions by immutable commit SHA where practical, and let Renovate handle
intentional pin updates.
