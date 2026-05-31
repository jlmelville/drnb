# drnb Maintenance

This document records the repository maintenance contract: workspace layout, Python policy,
lockfile usage, dependency upgrade order, and CI scope.

## Workspace Matrix

Core and SDK workspaces:

| Workspace | Python requirement | Lockfile | SDK dependency | Install and CI intent |
| --- | --- | --- | --- | --- |
| `.` | `>=3.13,<3.14` | `uv.lock` | `drnb-plugin-sdk-313`, `drnb-nn-plugin-sdk-313` | Strict install and full root checks on Python 3.13. |
| `plugin-sdks/drnb-plugin-sdk-312` | `>=3.12,<3.13` | `plugin-sdks/drnb-plugin-sdk-312/uv.lock` | None | SDK for plugin workspaces not yet migrated to Python 3.13. |
| `plugin-sdks/drnb-plugin-sdk-313` | `>=3.13,<3.14` | `plugin-sdks/drnb-plugin-sdk-313/uv.lock` | None | Default SDK for new core-hosted Python 3.13 plugin work. |
| `plugin-sdks/drnb-plugin-sdk-310` | `==3.10.14` | `plugin-sdks/drnb-plugin-sdk-310/uv.lock` | None | Legacy SDK for `ncvis`; lock check and targeted tests. |
| `nn-plugin-sdks/drnb-nn-plugin-sdk-312` | `>=3.12,<3.13` | `nn-plugin-sdks/drnb-nn-plugin-sdk-312/uv.lock` | None | SDK for NN plugin workspaces not yet migrated to Python 3.13. |
| `nn-plugin-sdks/drnb-nn-plugin-sdk-313` | `>=3.13,<3.14` | `nn-plugin-sdks/drnb-nn-plugin-sdk-313/uv.lock` | None | Default SDK for new core-hosted Python 3.13 NN plugin work. |

Embedder plugin workspaces:

| Workspace | Python requirement | SDK dependency | Lockfile | CI intent |
| --- | --- | --- | --- | --- |
| `plugins/pacmap` | `>=3.13,<3.14` | `drnb-plugin-sdk-313` | `plugins/pacmap/uv.lock` | Lock check and host-launched smoke on Python 3.13. |
| `plugins/cne` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/cne/uv.lock` | Lock check; GPU and PyKeOps behavior is manual. |
| `plugins/ncvis` | `==3.10.14` | `drnb-plugin-sdk-310` | `plugins/ncvis/uv.lock` | Legacy/manual; keep Python 3.10 and NumPy <2 until proven otherwise. |
| `plugins/pymde` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/pymde/uv.lock` | Lock check; Torch/GPU behavior is manual. |
| `plugins/topometry` | `>=3.12` | `drnb-plugin-sdk-312` | `plugins/topometry/uv.lock` | Deferred to dependency-upgrade review; current package upgrade is substantial. |
| `plugins/trimap` | `>=3.13,<3.14` | `drnb-plugin-sdk-313` | `plugins/trimap/uv.lock` | Lock check and host-launched smoke on Python 3.13. |
| `plugins/tsne` | `>=3.13,<3.14` | `drnb-plugin-sdk-313` | `plugins/tsne/uv.lock` | Lock check and host-launched smoke on Python 3.13. |
| `plugins/umato` | `>=3.13,<3.14` | `drnb-plugin-sdk-313` | `plugins/umato/uv.lock` | Lock check and host-launched smoke on Python 3.13. |

Nearest-neighbor plugin workspaces:

| Workspace | Python requirement | SDK dependency | Lockfile | CI intent |
| --- | --- | --- | --- | --- |
| `nn-plugins/annoy` | `>=3.13,<3.14` | `drnb-nn-plugin-sdk-313` | `nn-plugins/annoy/uv.lock` | Lock check and host-launched smoke on Python 3.13. |
| `nn-plugins/faiss` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/faiss/uv.lock` | Lock check only; FAISS itself is installed manually. |
| `nn-plugins/hnsw` | `>=3.13,<3.14` | `drnb-nn-plugin-sdk-313` | `nn-plugins/hnsw/uv.lock` | Lock check and host-launched smoke on Python 3.13. |
| `nn-plugins/torchknn` | `>=3.12` | `drnb-nn-plugin-sdk-312` | `nn-plugins/torchknn/uv.lock` | Lock check; GPU and PyTorch build behavior is manual. |

## Python Policy

The root package targets Python 3.13. The 3.13 SDKs intentionally require NumPy 2.1+ so CPython
3.13 syncs use wheels instead of attempting a NumPy 2.0 source build. Plugins should migrate to
Python 3.13 selectively after the core package validates. Do not move `plugins/ncvis` or
`plugin-sdks/drnb-plugin-sdk-310` away from Python 3.10 during routine cleanup.

Each workspace's `pyproject.toml` is the source of truth for Python compatibility. `.python-version`
files are only local interpreter-selection hints for pyenv users. Prefer major/minor values such as
`3.12`, `3.13`, or `3.10` unless a specific patch is known to be required. Keep the root
`.python-version` as the default developer interpreter, and add local `.python-version` files only
for true overrides such as Python 3.12 plugin workspaces or `ncvis`. When the repository's default
Python minor changes, update `AGENTS.md`, README setup guidance, this policy, SDK names, and
scaffolder defaults in the same milestone so agents and maintainers do not silently steer the repo
back to an older minor.

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
- Any caller-activated virtual environment is ignored for internal `uv` calls so each workspace
  uses its own `.venv`.
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

The checked-in GitHub Actions workflow is `.github/workflows/ci.yml`. It covers what the repo can
currently guarantee on hosted Ubuntu runners:

- Root Python 3.13.13 checks: `uv sync --locked`, `uv lock --check`,
  `uv run --locked ruff check .`, and `uv run --locked pytest` with `DRNB_HOME`
  set to a temporary runner directory.
- SDK checks for `plugin-sdks/drnb-plugin-sdk-310`,
  `plugin-sdks/drnb-plugin-sdk-312`, `plugin-sdks/drnb-plugin-sdk-313`,
  `nn-plugin-sdks/drnb-nn-plugin-sdk-312`, and
  `nn-plugin-sdks/drnb-nn-plugin-sdk-313`.
- Selected lightweight plugin checks for `plugins/pacmap`, `plugins/trimap`, `plugins/tsne`,
  `plugins/umato`, `nn-plugins/annoy`, and `nn-plugins/hnsw`.
- Plugin checks run `uv lock --check`, `uv sync --locked`, and a runner import smoke through
  `drnb-plugin-run.py --help` or `drnb-nn-plugin-run.py --help`. The PaCMAP workspace also runs
  its focused pytest suite.

Initial CI intentionally excludes:

- `nn-plugins/faiss`, because FAISS installation is manual/local and not representative of hosted
  runners.
- Torch-backed paths, including `plugins/pymde`, `plugins/cne`, and `nn-plugins/torchknn`, because
  CPU, CUDA, MPS, and old-GPU support need a separate policy.
- `plugins/ncvis`, because it remains a legacy Python 3.10 / NumPy <2 native build.
- `plugins/topometry`, because its package upgrade is deferred to a separate algorithm-plugin
  review.

Validate workflow changes locally before merging:

```bash
actionlint .github/workflows/*.yml
uvx zizmor .github/workflows
```

Third-party GitHub Actions are pinned by immutable commit SHA with the source tag in a comment.
When updating those pins, use `git ls-remote --tags` to resolve the intended tag to a commit SHA,
then rerun the workflow validation commands above.
