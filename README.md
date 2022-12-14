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

I tried stepping into the modern age with [poetry](https://python-poetry.org/) but had trouble with
[llvmite](https://pypi.org/project/llvmlite/) and [ncvis](https://pypi.org/project/ncvis/). So
pip it is: (of course a virtual env is highly recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
```

then from the base directory:

```bash
pip install -e .
```

or

```bash
pip install -e .[dev]
```

The `dev` extra identifier just installs some linting tools for use when developing `drnb` . If you
are using VSCode then the `.vscode/settings.json` sets those tools up with the configuration in
`setup.cfg` . That all assumes you are using the virtual environment in `venv`. Otherwise the usual
`pip install -e .` will do fine.

### faiss

If you have a GPU, I strongly recommend installing [faiss-gpu](https://pypi.org/project/faiss-gpu/)
for calculating exact nearest neighbors with the euclidean or cosine metric. That said, the
PyPI version is an [unofficially built wheel](https://github.com/facebookresearch/faiss/issues/1101)
and is currently stuck on version 1.7.2 due to
[the wheel size being too large](https://github.com/kyamagu/faiss-wheels/issues/57). Right now there
is a `pip install -e .[faiss_gpu]` identifier, but it doesn't do anything more than
`pip install faiss-gpu`. Alternatively, I was able to
[build my own version of GPU-powered faiss](https://gist.github.com/jlmelville/9b4f0d91ede13bff18d26759140709f9).

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

See the notebooks in `notebooks/embed-pipeline` .

### Notebook use

Code is all in the `drnb` module. Probably the following is a good chunk to stick at the top of
most notebooks (you can find it in `notebooks/template.ipynb` )

```python
%load_ext lab_black
import pandas as pd
import numpy as np
import drnb as nb
```

`lab_black` is an extension that runs the [black](https://black.readthedocs.io/en/stable/)
code formatter on notebook code on submit.
