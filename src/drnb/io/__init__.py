# Functions for reading and writing data

import bz2
import gzip
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import pandas as pd

from drnb.log import log
from drnb.preprocess import numpyfy
from drnb.util import islisty

DRNB_HOME_ENV_VAR = "DRNB_HOME"
DEBUG = False


def get_drnb_home(fail_if_not_set: bool = True) -> Optional[Path]:
    if DRNB_HOME_ENV_VAR in os.environ:
        return Path(os.environ[DRNB_HOME_ENV_VAR])
    if fail_if_not_set:
        raise ValueError("Environment variable {DRNB_HOME_ENV_VAR} not set")
    return None


def data_relative_path(path):
    if not DEBUG and path.is_relative_to(get_drnb_home()):
        return path.relative_to(get_drnb_home())
    return path


def stringify_paths(paths):
    return [str(data_relative_path(path)) for path in paths]


def get_path(drnb_home=None, sub_dir=None, create_sub_dir=False, verbose=False):
    if drnb_home is None:
        drnb_home = get_drnb_home()
        if drnb_home is None:
            raise ValueError(
                "No default path provided: " + f"set envvar {DRNB_HOME_ENV_VAR}"
            )
        if not drnb_home.is_dir():
            raise ValueError(f"Data root is not a directory: {str(drnb_home)}")
    data_path = drnb_home
    if sub_dir is not None:
        data_path = drnb_home / sub_dir
    if not data_path.exists():
        if create_sub_dir:
            if verbose:
                log.info(
                    "Directory %s does not exist, creating...",
                    data_relative_path(data_path),
                )
            data_path.mkdir(parents=False, exist_ok=False)
        else:
            raise FileNotFoundError(f"Missing directory {data_path}")
    return data_path


def ensure_suffix(suffix, default_suffix: Optional[str] = ""):
    if default_suffix is None:
        default_suffix = ""
    if suffix is None:
        suffix = default_suffix
    if not suffix:
        return suffix
    if islisty(suffix):
        return "".join(s if s[0] in (".", "-", "_") else f"-{s}" for s in suffix)
    if not suffix[0] in (".", "-", "_"):
        suffix = f"-{suffix}"
    return suffix


def ensure_file_extension(filename, ext):
    # could be a Path
    if not isinstance(filename, str):
        filename = str(filename)
    if not ext.startswith("."):
        ext = f".{ext}"
    if not filename.endswith(ext):
        return f"{filename}{ext}"
    return filename


def get_data_file_path(
    name,
    ext,
    suffix=None,
    drnb_home=None,
    sub_dir: Optional[str] = None,
    create_sub_dir=True,
    verbose=False,
):
    drnb_home = get_path(drnb_home, sub_dir, create_sub_dir, verbose)
    suffix = ensure_suffix(suffix, sub_dir)
    if suffix is not None:
        name = f"{name}{suffix}"
    name = ensure_file_extension(name, ext)

    return drnb_home / name


def read_data(
    dataset,
    suffix=None,
    drnb_home=None,
    sub_dir="data",
    as_numpy=False,
    verbose=False,
):
    for reader_func in (read_npy, read_pickle, read_pandas_csv):
        try:
            data = reader_func(
                dataset,
                suffix=suffix,
                drnb_home=drnb_home,
                sub_dir=sub_dir,
                verbose=verbose,
            )
            if not isinstance(as_numpy, bool):
                data = numpyfy(data, dtype=as_numpy)
            elif as_numpy:
                data = numpyfy(data)
            return data
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"Data for {dataset} suffix={suffix} sub_dir={sub_dir}")


def read_npy(name, suffix=None, drnb_home=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        ext="npy",
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info("Looking for npy format from %s", data_relative_path(data_file_path))
    return np.load(data_file_path)


def get_pkl_ext(compression=None):
    ext = ".pkl"
    if compression is not None:
        if compression == "":
            ext = ".pkl"
        elif compression == "gzip":
            ext = ".pkl.gz"
        elif compression == "bz2":
            ext = ".pkl.bz2"
        else:
            raise ValueError(f"Unknown compression type: {compression}")
    return ext


def read_pickle(
    name, suffix=None, drnb_home=None, sub_dir=None, verbose=False, compression="any"
):
    if compression == "any":
        compression = ["bz2", "gzip", ""]
    if not isinstance(compression, list):
        compression = [compression]
    exts = [get_pkl_ext(c) for c in compression]
    data_file_path = None
    for ext in exts:
        data_file_path = get_data_file_path(
            name,
            ext,
            suffix=suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=False,
            verbose=verbose,
        )
        if not data_file_path.exists():
            continue
        if verbose:
            log.info(
                "Looking for pkl format from %s", data_relative_path(data_file_path)
            )
        if ext == ".pkl":
            with open(data_file_path, "rb") as f:
                return pickle.load(f)
        elif ext == ".pkl.gz":
            with gzip.open(data_file_path, "rb") as f:
                return pickle.load(f)
        else:
            with bz2.open(data_file_path, "rb") as f:
                return pickle.load(f)
    if data_file_path is None:
        raise ValueError("No compression specified")
    raise FileNotFoundError(
        f"Missing pickle file {data_file_path.with_suffix('')} "
        f"for compression {compression}"
    )


def read_pandas_csv(name, suffix=None, drnb_home=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        "csv",
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info(
            "Looking for pandas csv format from %s", data_relative_path(data_file_path)
        )

    data = pd.read_csv(data_file_path, header=None)
    if np.any(data.dtypes.apply(pd.api.types.is_object_dtype)):
        header = 0
        if verbose:
            log.info("csv may have a header, retrying...")
        return pd.read_csv(data_file_path, header=header)
    return data


def read_json(name, suffix=None, drnb_home=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        ".json",
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )

    if verbose:
        log.info("Reading %s", data_relative_path(data_file_path))
    with open(data_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Handles case when data is already loaded
# if x and y are separate, return them
# if x is a tuple of two items, then split them into x and y
def get_xy(x, y):
    if y is None and isinstance(x, tuple) and len(x) == 2:
        y = x[1]
        x = x[0]
    return x, y


def write_csv(
    x,
    name,
    suffix=None,
    drnb_home=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".csv", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing csv format to %s", data_relative_path(output_path))
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x.to_csv(output_path, header=True, index=False)
    elif hasattr(x, "dtype") and (x.dtype is np.dtype(object) or x.dtype.kind == "U"):
        np.savetxt(output_path, x, delimiter=",", fmt="%s")
    else:
        np.savetxt(output_path, x, delimiter=",")
    return output_path


def write_npy(
    x,
    name,
    suffix=None,
    drnb_home=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".npy", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing numpy format to %s", data_relative_path(output_path))
    np.save(output_path, x)
    return output_path


def write_pickle(
    x,
    name,
    suffix=None,
    drnb_home=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
    compression=None,
    overwrite=True,
):
    ext = get_pkl_ext(compression)
    output_path = get_data_file_path(
        name, ext, suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing pkl format to %s", data_relative_path(output_path))
    if not overwrite and output_path.exists():
        raise FileExistsError(f"File {output_path} already exists")
    if compression is None:
        with open(output_path, "wb") as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    elif compression == "gzip":
        with gzip.open(output_path, "wb") as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    else:
        with bz2.open(output_path, "wb") as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    return output_path


def write_json(
    x,
    name,
    suffix=None,
    drnb_home=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".json", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing JSON format to %s", data_relative_path(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        if hasattr(x, "to_json"):
            f.write(x.to_json(indent=2))
        else:
            # here goes nothing
            f.write(json.dumps(x, indent=2, ensure_ascii=False))
    return output_path


def is_file_type(target_file_type, file_type=None, suffix=None):
    return (file_type is not None and file_type == target_file_type) or (
        suffix is not None and suffix.endswith(f".{target_file_type}")
    )


def write_data(
    x,
    name,
    suffix=None,
    drnb_home=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
    file_type: str | List[str] = "csv",
):
    if isinstance(file_type, str):
        file_type = [file_type]
    file_type = cast(List[str], file_type)
    suffix = ensure_suffix(suffix)

    output_paths = []
    for ftype in file_type:
        if is_file_type("csv", ftype, suffix):
            func = write_csv
        elif is_file_type("pkl", ftype, suffix):
            func = write_pickle
        elif is_file_type("npy", ftype, suffix):
            func = write_npy
        else:
            raise ValueError(f"Could not detect type of {ftype} to export to")

        output_path = func(
            x=x,
            name=name,
            suffix=suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
        )
        output_paths.append(output_path)
    return output_paths


@dataclass
class FileExporter:
    drnb_home: Optional[str] = None
    sub_dir: Optional[str] = None
    suffix: Optional[str] = None
    create_sub_dir: bool = True
    verbose: bool = False
    file_type: str = "csv"

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def export(self, name, data, suffix=None, sub_dir=None, drnb_home=None):
        if drnb_home is None:
            drnb_home = self.drnb_home
        if suffix is None:
            suffix = self.suffix
        if sub_dir is None:
            sub_dir = self.sub_dir
        output_path = write_data(
            data,
            name,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            suffix=suffix,
            create_sub_dir=self.create_sub_dir,
            verbose=self.verbose,
            file_type=self.file_type,
        )
        return output_path
