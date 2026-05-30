# drnb

Dimensionality reduction notebook tools.

Functions I used for quickly generating 2D dimensionality reduction plots and exporting data to
CSV. Very little reason for anyone to be interested in this, but it's too big for a gist. The idea
is that various pre-processing, nearest neighbors and other basic diagnostics about the data are
generated ahead of time and then where possible re-used to speed up calculations. Also, some basic
evaluation and attempts at simple visualization is used. Plus some home-brewed functions to (where
possible) provide some consistent initialization.

Eternally a work in progress: but there are some example notebooks that download and process
various datasets and that might be of interest to others.

## Installing

*November 29 2025* The Plugin Update

The problem that has bedeviled this repo has been too many dependencies. The current layout uses a
plugin architecture: one core project does most of the work, and separate plugin projects provide
embedding methods and nearest-neighbor backends. Communication between the core and the plugins is
by IPC, i.e. shelling out and running a Python script in each plugin folder. Plugin SDKs provide
helper functions for reading and writing requests and responses.

The main outcome is that installing can be either a narrow core install or a full repository install.

### Setup

You will need [uv](https://docs.astral.sh/uv/) and Python version management through uv or
[pyenv](https://github.com/pyenv/pyenv). The root project currently targets Python 3.12.

For the core package only, run:

```bash
uv sync --locked
```

This installs the core package and the Python 3.12 plugin SDK packages used by the host. It does
not install every external plugin environment.

For a full repository install, run:

```bash
./scripts/install.sh
```

The script installs the SDK workspaces and core package as strict steps, then installs embedder and
nearest-neighbor plugins as best effort. It uses checked-in lockfiles by default. A best-effort
plugin failure is reported in the final summary but does not abort the whole install, because some
plugins depend on native packages, old Python versions, GPU-specific PyTorch builds, or manual local
setup.

If you need to make changes to one of the plugins (e.g. adjusting the version of `pytorch`), then
run `./scripts/install.sh --reinstall-all` to reinstall all plugin packages without requiring SDK
version bumps. To target a single plugin, use `./scripts/install.sh --reinstall <name>`.

To deliberately refresh lockfiles during maintenance, use `./scripts/install.sh --refresh-locks`.

Useful development checks:

```bash
uv lock --check
uv run ruff check .
uv run pytest
```

See `docs/maintenance.md` for the workspace matrix, lockfile policy, dependency upgrade buckets,
and planned CI scope. See `docs/plugin-protocol.md` and `docs/nn-plugin-protocol.md` for the plugin
runner contracts.

### Optional packages

#### Faiss

With a recent CUDA update to
12.2, I have finally successfully [built Faiss with GPU support on WSL2 with Ubuntu](https://gist.github.com/jlmelville/9b4f0d91ede13bff18d26759140709f9)
and a Pascal-era card (GTX 1080). Unfortunately, generating nearest neighbors is a lot slower
without it.

#### ncvis

Currently does not work on ARM Macs. It is intentionally isolated on Python 3.10 with NumPy <2, so
it should not be treated as a required install on every machine.

## Data setup

Before running anything, set a home directory for drnb, `DRNB_HOME` . Datasets will be imported to,
and embedding results exported to, folders underneath this directory, e.g.:

```bash
export DRNB_HOME=~/dev/drnbhome
```

## Documentation

### Importing data

See the notebooks in `notebooks/data-pipeline` for how to get data processed and into a place where
`drnb` can use it. Apart from cleaning and saving the numerical features to be processed, metadata
used for labeling and coloring plots is specified, and nearest neighbor and triplet data is
pre-calculated.

### Embedding

See the notebooks in `notebooks/embed-pipeline` for example output from most of the embedding
methods supported. `notebooks/experiments.ipynb` demonstrates how to compare multiple methods
against different datasets. `notebooks/plot-options.ipynb` provides information on controlling plot
output and extra diagnostic plots.

### Using plotly

For the plotly charts, you may need to install the `jupyterlab-plotly` extension, which also
requires node to be installed:

```bash
jupyter labextension install jupyterlab-plotly
```

If not installed, plotly charts will display as long as you set
`clickable=False, renderer="iframe"`. You miss out on being able to create custom click and hover
handlers.

That said, as of 2025 I use the VSCode support for notebook rendering and it didn't need special
treatment beyond installing the dependencies in the `pyproject.toml`.
