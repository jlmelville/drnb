import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_PATH = Path.home() / "rdev" / "datasets"
DATA_PATH = DATASETS_PATH / "py"


def read_data(dataset, suffix=None, data_path=None, repickle=False, header=None):
    if data_path is None:
        data_path = DATA_PATH
    if suffix is not None:
        dataset_basename = f"{dataset}-{suffix}"
    else:
        dataset_basename = dataset
    pickle_name = f"{dataset_basename}.pickle"
    pickle_path = data_path / pickle_name

    if pickle_path.exists() and not repickle:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        csv_name = f"{dataset_basename}.csv"
        csv_path = data_path / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        data = pd.read_csv(csv_path, header=header)
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return data


def read_dataxy(dataset, data_path=None, repickle=False):
    header = None
    x = read_data(dataset, data_path=data_path, repickle=repickle, header=header)
    if np.any(x.dtypes.apply(pd.api.types.is_object_dtype)):
        header = 0
        print("X may have a header, retrying...")
        x = read_data(dataset, data_path=data_path, repickle=True, header=header)
    try:
        y = read_data(
            dataset, "y", data_path=data_path, repickle=repickle, header=header
        )
    except FileNotFoundError:
        y = range(x.shape[0])
    return (x, y)


def export_coords(embedded, name, export_dir, suffix=None):
    if isinstance(embedded, tuple):
        coords = embedded[0]
    else:
        coords = embedded
    if suffix is None:
        suffix = export_dir
    if not suffix[0] in ("-", "_"):
        suffix = f"-{suffix}"
    write_csv(coords, name=f"{name}{suffix}", sub_dir=export_dir)


def get_xy_data(name, x=None, y=None, repickle=False):
    if x is None:
        x, y = read_dataxy(name, repickle=repickle)
    if y is None:
        y = range(x.shape[0])
    return x, y


def get_xy(x, y):
    if y is None and isinstance(x, tuple) and len(x) == 2:
        y = x[1]
        x = x[0]
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return x, y


def write_csv(x, name, data_path=None, sub_dir=None):
    if data_path is None:
        data_path = DATASETS_PATH
    if sub_dir is not None:
        data_path = data_path / sub_dir
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    output_path = data_path / name
    if x.dtype is np.dtype(object) or x.dtype.kind == "U":
        np.savetxt(output_path, x, delimiter=",", fmt="%s")
    else:
        np.savetxt(output_path, x, delimiter=",")
