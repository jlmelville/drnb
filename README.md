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

The problem that has bedeviled this repo has been too many dependencies. I have taken the nuclear
option and created a "plugin" architecture: instead of one project, there is now one "core"
project that does most of the work, and then several separate projects, one for each embedding
method and nearest neighbor package. Communication between the core and the plugins is just by IPC,
i.e. shelling out and running a python script in each plugin folder. There are also plugin SDKs
to provide useful helper functions for reading and writing the request and responses.

The main outcome is that installing is now a bit more involved because you must recursively install
several packages in this repo.

### Setup

You will need [uv](https://docs.astral.sh/uv/) and [pyenv](https://github.com/pyenv/pyenv) to
handle installation.

Run the script:

```bash
./scripts/install.sh
```

to install everything. This will go through all the different projects and install them into
virtual environments. Most packages should install ok but some are troublesome, so it shouldn't be
a failure if some packages fail to install. Some may require different versions of python from the
drnb core, which is where pyenv comes in.

If you need to make changes to one of the plugins (e.g. adjusting the version of `pytorch`), then
I recommend running `./scripts/install.sh -a` to reinstall all packages to avoid problems with not
adjusting the version string.

After installing, you can just usually work with the core of drnb (and the notebooks) with:

```bash
uv venv
source .venv/bin/activate
uv sync
# or if you want:
# uv sync --dev
```

The `dev` extra identifier just installs some linting tools for use when developing `drnb` . If you
are using VSCode then the `.vscode/settings.json` sets those tools up. I am trying to see how far
I can get with just [ruff](https://docs.astral.sh/ruff/).

See the `docs` folder for more details on the different plugins and the SDK they follow.

### Optional packages

#### Faiss

With a recent CUDA update to
12.2, I have finally successfully [built Faiss with GPU support on WSL2 with Ubuntu](https://gist.github.com/jlmelville/9b4f0d91ede13bff18d26759140709f9)
and a Pascal-era card (GTX 1080). Unfortunately, generating nearest neighbors is a lot slower
without it.

#### ncvis

Currently does not work on ARM Macs. It shouldn't be a failure to fail to install this.

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

For the plotly charts, install the `jupyterlab-plotly` extension, which also requires node to
be installed:

```bash
jupyter labextension install jupyterlab-plotly
```

If not installed, plotly charts will display as long as you set
`clickable=False, renderer="iframe"`. You miss out on being able to create custom click and hover
handlers.
