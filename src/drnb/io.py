# Functions for reading and writing data

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path.home() / "rdev" / "datasets"


def read_data(
    dataset, suffix=None, data_path=None, sub_dir="xy", repickle=False, header=None
):
    if data_path is None:
        data_path = DATA_ROOT
    if sub_dir is not None:
        data_path = data_path / sub_dir
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
    return x, y


class XImporter:
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    # pylint: disable=unused-argument
    def import_data(self, name, x, y):
        return get_xy(x, y)


@dataclass
class DatasetImporter:
    repickle: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def import_data(self, name, x, y):
        x, y = get_xy_data(name, x=None, y=None, repickle=self.repickle)
        return x, y


def export_coords(
    embedded,
    name,
    export_dir,
    data_path=None,
    suffix=None,
    create_sub_dir=False,
    verbose=False,
):
    if isinstance(embedded, tuple):
        coords = embedded[0]
    else:
        coords = embedded
    if suffix is None:
        suffix = export_dir
    if not suffix[0] in ("-", "_"):
        suffix = f"-{suffix}"
    write_csv(
        coords,
        name=f"{name}{suffix}",
        data_path=data_path,
        sub_dir=export_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
    )


def write_csv(
    x, name, data_path=None, sub_dir=None, create_sub_dir=False, verbose=False
):
    if data_path is None:
        data_path = DATA_ROOT
    if sub_dir is not None:
        data_path = data_path / sub_dir
    if not data_path.exists():
        if create_sub_dir:
            if verbose:
                print(f"{data_path} does not exist, creating...")
            data_path.mkdir(parents=False, exist_ok=False)
        else:
            raise FileNotFoundError(f"Missing directory {data_path}")
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    output_path = data_path / name
    if x.dtype is np.dtype(object) or x.dtype.kind == "U":
        np.savetxt(output_path, x, delimiter=",", fmt="%s")
    else:
        np.savetxt(output_path, x, delimiter=",")


class NoExporter:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def export(self, name, coords):
        pass


@dataclass
class CsvExporter:
    export_dir: str = None
    data_path: str = None
    suffix: str = None
    create_sub_dir: bool = False
    verbose: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def export(self, name, coords):
        export_coords(
            coords,
            name,
            export_dir=self.export_dir,
            data_path=self.data_path,
            suffix=self.suffix,
            create_sub_dir=self.create_sub_dir,
            verbose=self.verbose,
        )
