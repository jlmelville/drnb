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

### The data root directory

Defaults for the functions assume that there is a root directory under which sub-directories store
input and output data. The root directory is stored in `nb.io.DATA_ROOT` is set to a value of 
convenience to me. If you are not me, you will want to set:

```python
from pathlib import Path

nb.io.DATA_ROOT = Path.home() / "the-base-directory-where-your-data-lives"
```

### Input data

The input datasets of interest are stored as files in a sub-directory called (by default) `xy`:

```text
DATA_ROOT/
  xy/
    iris.csv
    iris-y.csv
    mnist.csv
    mnist-y.csv
    fashion.pickle
    fashion-y.pickle
```

The dataset name if the basename of the file, e.g. the contents of `iris.csv` `iris-y.csv` are 
loaded in when using the dataset name `iris` with the various `dnrb` function. The `-y` files are 
optional CSV files with categorical labels or real values. For datasets without y-labels, an array
of 0..N is used for the purpose of coloring any produced plots.

As shown with `fashion.pickle` and `fashion-y.pickle`, data can also be stored as a pickle file. In
fact, data will be automatically re-written to pickle format and if exists, it is read in preference
to the raw CSV files for speed purposes (after the initial cost of conversion and writing to disk).

No datasets are in this repo. You will have so source them for yourself.

## Examples

### Reading

```python
import drnb.io as nbio

irisx, irisy = nbio.read_dataxy("iris")
```

These are read in as [pandas](https://pandas.pydata.org/) dataframes. You can override `DATA_ROOT`
and the input directory name here:

```python
# change data_path and sub_dir to something more to your liking
irisx, irisy = nb.read_dataxy("iris", data_path=nb.io.DATA_ROOT, sub_dir="xy")
```

### Plotting

This doesn't make much sense to do outside of the notebook.

`sns_embed_plot` takes embedded coordinates and uses [seaborn](https://seaborn.pydata.org/) to come
up with a scatterplot of the results, employing labels provided in a Pandas dataframe. Continuing
from above:

```python
from sklearn.random_projection import SparseRandomProjection
import drnb.plot as nbplot

transformer = SparseRandomProjection(n_components=2, random_state=42)
iris_randproj = transformer.fit_transform(irisx)

nbplot.sns_embed_plot(iris_randproj, irisy)
```

### Writing

Mainly I care about exporting to CSV so it's easy to read in with other tools or languages. The
output convention is similar to the input. For the example above, I would want the datasets
to have a suffix `-randproj.csv` and to all live in a folder called `randproj`:

```text
DATA_ROOT/
  randproj/
    iris-randproj.csv
    mnist-randproj.csv
```

This is accomplished with:

```python
nbio.export_coords(iris_randproj, name="iris", export_dir="randproj")
```

You can also over-ride `data_root` here too.

### Read/embed/write in one step

All of the above example can be run in one go with:

```python
import drnb

iris_randproj = nb.embed_data(
    name="iris",
    method=("randproj", dict(random_state=42)),
    export=dict(create_sub_dir=True),
)
```

The coordinates are returned in case you want to do something with them. If you don't want to pass
any extra options to the `method`, you can just pass the name of the method directly, e.g.
`method="randproj"`.

#### Writing extra data

The embedding method will usually return a matrix of the coordinates. If more data has been
specified to return, a `dict` is returned instead with the coordinates under the `coords` key. Other
data is stored under a key that will be used as a further suffix if exporting is to be done:

```python
iris_densmap = nb.embed_data(
    "iris",
    method=("densmap", dict(output_dens=True))
    export=dict(export_dir="dmaprad"),
)
iris_densmap.keys()
```

```python
dict_keys(['coords', 'dens_ro', 'dens_re'])
```

Instead of exporting to the `densmap` directory, the file layout is:

```text
DATA_ROOT/
  dmaprad/
    iris-dmaprad.csv
    iris-dmaprad-dens_ro.csv
    iris-dmaprad-dens_re.csv
```

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

