"""Functions for reading and writing data in the data repository."""

import bz2
import gzip
import json
import os
import pickle
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

from drnb.log import log
from drnb.preprocess import numpyfy

DRNB_HOME_ENV_VAR = "DRNB_HOME"
DEBUG = False


def get_drnb_home_maybe() -> Path | None:
    """Get the root directory for the data repository. If the environment variable
    DRNB_HOME is set, return the path. Otherwise return None."""
    if DRNB_HOME_ENV_VAR in os.environ:
        return Path(os.environ[DRNB_HOME_ENV_VAR])
    return None


def get_drnb_home() -> Path:
    """Get the root directory for the data repository. If the environment variable
    DRNB_HOME is set, return the path. Otherwise, raise an error."""
    if DRNB_HOME_ENV_VAR in os.environ:
        return Path(os.environ[DRNB_HOME_ENV_VAR])
    raise ValueError(f"Environment variable {DRNB_HOME_ENV_VAR} not set")


def data_relative_path(path: Path) -> Path:
    """Return the relative path of a file or directory within the data repository."""
    drnb_home = get_drnb_home()
    if not DEBUG and path.is_relative_to(drnb_home):
        return path.relative_to(drnb_home)
    return path


def stringify_paths(paths: Iterable[Path]) -> list[str]:
    """Convert a list of Path objects to a list of strings."""
    return [str(data_relative_path(path)) for path in paths]


def get_path(
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = False,
    verbose: bool = False,
) -> Path:
    """Get the path to the data repository. If drnb_home is provided, return it. If not,
    get the path from the environment variable DRNB_HOME. If that is not set, raise an
    error if fail_if_not_set is True, otherwise return None. If sub_dir is provided, get
    the path to the subdirectory within the data repository. If the subdirectory does not
    exist, create it if create_sub_dir is True, otherwise raise an error."""
    if drnb_home is None:
        drnb_home = get_drnb_home()
        if drnb_home is None:
            raise ValueError(
                f"No default path provided: set envvar {DRNB_HOME_ENV_VAR}"
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


def ensure_suffix(
    suffix: str | list[str] | None, default_suffix: str | list[str] | None = ""
) -> str:
    """Ensure that the suffix is a string starting with a hyphen or period. If suffix is
    None, return default_suffix. If suffix is an empty string, return it. If suffix is a
    list of strings, return the concatenation of the strings with a hyphen between them.
    If suffix is a string that does not start with a hyphen or period, add a hyphen to
    the beginning of the string."""
    if default_suffix is None:
        default_suffix = ""
    if suffix is None:
        suffix = default_suffix
    if suffix == "":
        return suffix
    if isinstance(suffix, (list, tuple)):
        return "".join(s if s[0] in (".", "-", "_") else f"-{s}" for s in suffix)
    if not suffix[0] in (".", "-", "_"):
        suffix = f"-{suffix}"
    return suffix


def ensure_file_extension(filename: Path | str, ext: str) -> str:
    """Ensure that the filename has the given extension. If the filename does not have
    the extension, add it. If the extension does not start with a period, add it."""
    # could be a Path
    if not isinstance(filename, str):
        filename = str(filename)
    if not ext.startswith("."):
        ext = f".{ext}"
    if not filename.endswith(ext):
        return f"{filename}{ext}"
    return filename


def get_data_file_path(
    name: str,
    ext: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Get the path to a file in the data repository. The file name is constructed from
    the name, suffix, and extension. The file is located in the subdirectory of the data
    repository specified by sub_dir. If the subdirectory does not exist, it is created
    if create_sub_dir is True, otherwise an error is raised. The path to the file is
    returned."""
    drnb_home = get_path(drnb_home, sub_dir, create_sub_dir, verbose)
    suffix = ensure_suffix(suffix, sub_dir)
    name = f"{name}{suffix}"
    name = ensure_file_extension(name, ext)

    return drnb_home / name


def read_data(
    dataset: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str = "data",
    as_numpy: bool | DTypeLike = False,
    verbose: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Read data from the data repository. The data is read from a numpy file, a pickle
    file, or a CSV file, depending on which is found first. The data is returned as a
    numpy array or pandas DataFrame. If as_numpy is True, the data is converted to a
    numpy array before being returned. If as_numpy is a dtype, the data is converted to
    a numpy array with the given dtype before being returned. If the data is not found,
    a FileNotFoundError is raised."""
    for reader_func in (
        read_npy,
        read_feather,
        read_parquet,
        read_pickle,
        read_pandas_csv,
    ):
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
        except ModuleNotFoundError as e:
            # Seems like this could happen if we read pickled dataframe based on older
            # pandas version
            log.warning("Module not found: %s from %s", e, reader_func.__name__)
        except ValueError as e:
            # usually from reading numpy array with object dtype
            log.warning("Value error: %s from %s", e, reader_func.__name__)
    raise FileNotFoundError(f"Data for {dataset} suffix={suffix} sub_dir={sub_dir}")


def read_npy(
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Read a numpy file from the data repository and return the data."""
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


def get_pkl_ext(compression: Literal["gzip", "bz2", ""] | None = None) -> str:
    """Get the file extension for a pickle file with the given compression type. If
    compression is None or an empty string, return ".pkl". If compression is "gzip",
    return ".pkl.gz". If compression is "bz2", return ".pkl.bz2". Otherwise, raise a
    ValueError."""
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
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
    compression: Literal["gzip", "bz2", "any", ""] | list[str] = "any",
) -> pd.DataFrame | np.ndarray | dict:
    """Read a pickle file from the data repository and return the data. The compression
    type can be specified as "gzip", "bz2", "any", or an empty string. If "any" is
    specified, the function will try to read the file with each compression type in
    turn. If the file is not found, a FileNotFoundError is raised."""
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


def read_pandas_csv(
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read a CSV file from the data repository and return it as a pandas DataFrame."""
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


def read_json(
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
) -> dict:
    """Read a JSON file from the data repository."""
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


def read_parquet(
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read a parquet file from the data repository and return it as a pandas DataFrame."""
    data_file_path = get_data_file_path(
        name,
        ".parquet",
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info(
            "Looking for parquet format from %s", data_relative_path(data_file_path)
        )

    return pd.read_parquet(data_file_path, engine="pyarrow")


def read_feather(
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read a feather file from the data repository and return it as a pandas DataFrame."""
    data_file_path = get_data_file_path(
        name,
        ".feather",
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info(
            "Looking for feather format from %s", data_relative_path(data_file_path)
        )

    return pd.read_feather(data_file_path)


def write_csv(
    x: pd.DataFrame | pd.Series | np.ndarray,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Write data to a CSV file in the data repository. The data can be a pandas
    DataFrame, a pandas Series, or a numpy array. The data is written to a file with the
    given name, suffix, and extension. The file is located in the subdirectory of the
    data repository specified by sub_dir. If the subdirectory does not exist, it is
    created if create_sub_dir is True, otherwise an error is raised. The path to the
    file is returned."""
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
    x: np.ndarray,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Write data to a numpy file in the data repository. The data is written to a file
    with the given name, suffix, and extension. The file is located in the subdirectory
    of the data repository specified by sub_dir. If the subdirectory does not exist, it
    is created if create_sub_dir is True, otherwise an error is raised. The path to the
    file is returned."""
    output_path = get_data_file_path(
        name, ".npy", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing numpy format to %s", data_relative_path(output_path))
    np.save(output_path, x)
    return output_path


def write_pickle(
    x: pd.DataFrame | pd.Series | np.ndarray,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
    compression: Literal["gzip", "bz2", ""] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write data to a pickle file in the data repository. The data can be a pandas
    DataFrame, a pandas Series, or a numpy array. The data is written to a file with the
    given name, suffix, and extension. The file is located in the subdirectory of the
    data repository specified by sub_dir. If the subdirectory does not exist, it is
    created if create_sub_dir is True, otherwise an error is raised. The file can be
    compressed with gzip or bz2. If overwrite is False, an error is raised if the file
    already exists. The path to the file is returned."""
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
    x: dict | Any,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Write data to a JSON file in the data repository. Success is not guaranteed
    unless the object is a dictionary or a dataclass instance.
    The data is written to a file with the given name, suffix, and extension. The file
    is located in the subdirectory of the data repository specified by sub_dir. If the
    subdirectory does not exist, it is created if create_sub_dir is True, otherwise an
    error is raised. The path to the file is returned."""
    output_path = get_data_file_path(
        name, ".json", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing JSON format to %s", data_relative_path(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        if is_dataclass(x):
            f.write(json.dumps(asdict(x), indent=2, ensure_ascii=False))
        else:
            # here goes nothing
            f.write(json.dumps(x, indent=2, ensure_ascii=False))
    return output_path


def write_parquet(
    x: pd.DataFrame,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Write data to a parquet file in the data repository. The data is written to a
    file with the given name, suffix, and extension. The file is located in the
    subdirectory of the data repository specified by sub_dir. If the subdirectory does
    not exist, it is created if create_sub_dir is True, otherwise an error is raised.
    The path to the file is returned."""
    output_path = get_data_file_path(
        name, ".parquet", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing parquet format to %s", data_relative_path(output_path))
    x.to_parquet(output_path, engine="pyarrow", compression="snappy", index=True)
    return output_path


def write_feather(
    x: pd.DataFrame,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
) -> Path:
    """Write data to a feather file in the data repository. The data is written to a
    file with the given name, suffix, and extension. The file is located in the
    subdirectory of the data repository specified by sub_dir. If the subdirectory does
    not exist, it is created if create_sub_dir is True, otherwise an error is raised.
    The path to the file is returned."""
    output_path = get_data_file_path(
        name, ".feather", suffix, drnb_home, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing feather format to %s", data_relative_path(output_path))
    x.to_feather(output_path)
    return output_path


def is_file_type(
    target_file_type: str,
    file_type: str | None = None,
    suffix: str | list[str] | None = None,
) -> bool:
    """Check if the file type matches the target file type or ends with the target file
    type. For example, if the target file type is "csv", compare to "csv" or ".csv"."""
    return (file_type is not None and file_type == target_file_type) or (
        suffix is not None and suffix.endswith(f".{target_file_type}")
    )


def write_data(
    x: pd.DataFrame | pd.Series | np.ndarray,
    name: str,
    suffix: str | list[str] | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str | None = None,
    create_sub_dir: bool = True,
    verbose: bool = False,
    file_type: str | list[str] = "csv",
) -> list[Path]:
    """Write data to one or more files in the data repository. The data can be a pandas
    DataFrame, a pandas Series, or a numpy array. The data is written to one or more
    files with the given name, suffix, and extension. The file is located in the
    subdirectory of the data repository specified by sub_dir. If the subdirectory does
    not exist, it is created if create_sub_dir is True, otherwise an error is raised.
    The file type can be one or more of "parquet", "csv", "pkl", or "npy". The paths to
    the files are returned."""
    if isinstance(file_type, str):
        file_type = [file_type]
    suffix = ensure_suffix(suffix)
    output_paths = []
    for ftype in file_type:
        if is_file_type("parquet", ftype, suffix):
            if not isinstance(x, pd.DataFrame):
                continue
            func = write_parquet
        elif is_file_type("feather", ftype, suffix):
            if not isinstance(x, pd.DataFrame):
                continue
            func = write_feather
        elif is_file_type("csv", ftype, suffix):
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
    """Class to export data to files in the data repository."""

    drnb_home: Path | str | None = None
    sub_dir: str | None = None
    suffix: str | list[str] | None = None
    create_sub_dir: bool = True
    verbose: bool = False
    file_type: str | list[str] = "csv"

    @classmethod
    def new(cls, **kwargs):
        """Create a new FileExporter object from the given keyword arguments.

        Arguments:
        - drnb_home: Path | str | None = None. The root directory for the data repository.
        - sub_dir: str | None = None. The subdirectory within the data repository.
        - suffix: str | list[str] | None = None. The suffix to add to the file name.
        - create_sub_dir: bool = True. Whether to create the subdirectory if it does not exist.
        - verbose: bool = False. Whether to print verbose output.
        - file_type: str | list[str] = "csv". The type of file or files to export to.
        """
        return cls(**kwargs)

    def export(
        self,
        name: str,
        data: pd.DataFrame | pd.Series | np.ndarray,
        suffix: str | list[str] | None = None,
        sub_dir: str | None = None,
        drnb_home: Path | str | None = None,
    ) -> list[Path]:
        """Export data to one or more files in the data repository. The data can be a
        pandas DataFrame, a pandas Series, or a numpy array. The data is written to one
        or more files with the given name, suffix, and extension. The file is located in
        the subdirectory of the data repository specified by sub_dir. The paths to
        the files are returned."""
        if drnb_home is None:
            drnb_home = self.drnb_home
        if suffix is None:
            suffix = self.suffix
        if sub_dir is None:
            sub_dir = self.sub_dir
        output_paths = write_data(
            data,
            name,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            suffix=suffix,
            create_sub_dir=self.create_sub_dir,
            verbose=self.verbose,
            file_type=self.file_type,
        )
        return output_paths
