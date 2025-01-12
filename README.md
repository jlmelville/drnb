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

*January 11 2025* Now using Python 3.12 and [uv](https://docs.astral.sh/uv/). Trying to use too
many dependencies makes things way too brittle, so fewer embedding methods are supported.

The `dev` extra identifier just installs some linting tools for use when developing `drnb` . If you
are using VSCode then the `.vscode/settings.json` sets those tools up. I am trying to see how far
I can get with just [ruff](https://docs.astral.sh/ruff/).

```bash
uv pip install -e .
uv export --only-group=dev | uv pip install --requirements=-
```

The second line is for installing the 'dev' dependency group (see
<https://github.com/astral-sh/uv/issues/8590>).

### Optional packages

I will be deprecating packages that are giving trouble.

#### Faiss

With a recent CUDA update to
12.2, I have finally successfully [built Faiss with GPU support on WSL2 with Ubuntu](https://gist.github.com/jlmelville/9b4f0d91ede13bff18d26759140709f9)
and a Pascal-era card (GTX 1080). Unfortunately, generating nearest neighbors is a lot slower
without it.

## Data setup

Before running anything, set a home directory for drnb, `DRNB_HOME` . Datasets will be imported to,
and embedding results exported to, folders underneath this directory, e.g.:

```bash
export DRNB_HOME=~/dev/drnbhome
```

## Importing data

See the notebooks in `notebooks/data-pipeline` for how to get data processed and into a place where
`drnb` can use it. Apart from cleaning and saving the numerical features to be processed, metadata
used for labeling and coloring plots is specified, and nearest neighbor and triplet data is
pre-calculated.

## Embedding

See the notebooks in `notebooks/embed-pipeline`.

### Using plotly

For the plotly charts, install the `jupyterlab-plotly` extension, which also requires node to
be installed:

```bash
jupyter labextension install jupyterlab-plotly
```

If not installed, plotly charts will display as long as you set
`clickable=False, renderer="iframe"`. You miss out on being able to create custom click and hover
handlers.
