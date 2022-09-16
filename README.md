# drnb

Dimensionality reduction notebook tools.

Functions I used for quickly generating 2D dimensionality reduction plots and exporting data to
CSV. Very little reason for anyone to be interested in this, but it's too big for a gist.

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
pip install -e .[dev]
```

The `dev` extra identifier just installs some linting tools for use when developing `drnb`. If you
are using VSCode then the `.vscode/settings.json` sets those tools up with the configuration in
`setup.cfg`. That all assumes you are using the virtual environment in `venv`. Otherwise the usual
`pip install -e .` will do fine.

## Data setup

Before running anything, set a home directory for drnb, `DRNB_HOME`. Datasets will be imported to,
and embedding results exported to, folders underneath this directory, e.g.:

```bash
export DRNB_HOME=~/dev/drnbhome
```

### Importing data

See the notebooks in `notebooks/data-pipeline` for how to get data processed and into a place where
`drnb` can use it. Apart from cleaning and saving the numerical features to be processed, metadata
used for labeling and coloring plots is specified, and nearest neighbor and triplet data is
pre-calculated.

## Notebook use

Code is all in the `drnb` module. Probably the following is a good chunk to stick at the top of
most notebooks (you can find it in `notebooks/template.ipynb`)

```python
%load_ext lab_black
import pandas as pd
import numpy as np
import drnb as nb
```

`lab_black` is an extension that runs the [black](https://black.readthedocs.io/en/stable/)
code formatter on notebook code on submit.
