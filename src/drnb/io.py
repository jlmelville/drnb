# Functions for reading and writing data

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drnb.log import log
from drnb.preprocess import numpyfy
from drnb.util import get_method_and_args, islisty

DATA_ROOT = Path.home() / "rdev" / "datasets"
DEBUG = False


def data_relative_path(path):
    if not DEBUG and path.is_relative_to(DATA_ROOT):
        return path.relative_to(DATA_ROOT)
    return path


def get_data_path(data_path=None, sub_dir=None, create_sub_dir=False, verbose=False):
    if data_path is None:
        data_path = DATA_ROOT
    if sub_dir is not None:
        data_path = data_path / sub_dir
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


def ensure_suffix(suffix, default_suffix=""):
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
    data_path=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    data_path = get_data_path(data_path, sub_dir, create_sub_dir, verbose)
    suffix = ensure_suffix(suffix, sub_dir)
    if suffix is not None:
        name = f"{name}{suffix}"
    name = ensure_file_extension(name, ext)

    return data_path / name


def read_data(
    dataset,
    suffix=None,
    data_path=None,
    sub_dir="xy",
    as_numpy=False,
    verbose=False,
):
    for reader_func in (read_npy, read_pickle, read_pandas_csv):
        try:
            data = reader_func(
                dataset,
                suffix=suffix,
                data_path=data_path,
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


def read_datax(dataset, data_path=None, sub_dir="xy", verbose=False):
    return read_data(
        dataset,
        suffix="",
        data_path=data_path,
        sub_dir=sub_dir,
        verbose=verbose,
        as_numpy=True,
    )


def read_datay(dataset, data_path=None, sub_dir="xy", verbose=False, x=None):
    try:
        y = read_data(
            dataset, suffix="y", data_path=data_path, sub_dir=sub_dir, verbose=verbose
        )
    except FileNotFoundError:
        if x is None:
            x = read_data(
                dataset, suffix="", data_path=data_path, sub_dir=sub_dir, verbose=False
            )
        y = range(x.shape[0])
    return y


def read_dataxy(dataset, data_path=None, sub_dir="xy", verbose=False):
    x = read_datax(dataset, data_path=data_path, sub_dir=sub_dir, verbose=verbose)
    y = read_datay(dataset, data_path=data_path, sub_dir=sub_dir, verbose=verbose, x=x)
    return x, y


def read_npy(name, suffix=None, data_path=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        ext="npy",
        suffix=suffix,
        data_path=data_path,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info("Looking for npy format from %s", data_relative_path(data_file_path))
    return np.load(data_file_path)


def read_pickle(name, suffix=None, data_path=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        "pickle",
        suffix=suffix,
        data_path=data_path,
        sub_dir=sub_dir,
        create_sub_dir=False,
        verbose=verbose,
    )
    if verbose:
        log.info(
            "Looking for pickle format from %s", data_relative_path(data_file_path)
        )
    with open(data_file_path, "rb") as f:
        return pickle.load(f)


def read_pandas_csv(name, suffix=None, data_path=None, sub_dir=None, verbose=False):
    data_file_path = get_data_file_path(
        name,
        "csv",
        suffix=suffix,
        data_path=data_path,
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


def list_available_data(data_path=None, sub_dir="xy", with_y=False):
    if data_path is None:
        data_path = DATA_ROOT
    if sub_dir is not None:
        data_path = data_path / sub_dir

    datasets = {x.stem for x in Path.glob(data_path, "*") if not x.stem.endswith("-y")}

    if with_y:
        # chop off the "-y" bit
        y_stems = {
            y.stem[:-2] for y in Path.glob(data_path, "*") if y.stem.endswith("-y")
        }
        # only keep items in datasets and y_stems
        datasets &= y_stems

    return sorted(datasets)


# Handles case when data is already loaded
# if x and y are separate, return them
# if x is a tuple of two items, then split them into x and y
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
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def import_data(self, name, x=None, y=None):
        x, y = read_dataxy(name)
        return x, y


def create_importer(x=None, import_kwargs=None):
    if x is None:
        importer_cls = DatasetImporter
    else:
        importer_cls = XImporter

    if import_kwargs is None:
        import_kwargs = {}

    importer = importer_cls.new(**import_kwargs)
    return importer


def write_csv(
    x,
    name,
    suffix=None,
    data_path=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".csv", suffix, data_path, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing csv format to %s", data_relative_path(output_path))
    if x.dtype is np.dtype(object) or x.dtype.kind == "U":
        np.savetxt(output_path, x, delimiter=",", fmt="%s")
    else:
        np.savetxt(output_path, x, delimiter=",")


def write_npy(
    x,
    name,
    suffix=None,
    data_path=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".npy", suffix, data_path, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing numpy format to %s", data_relative_path(output_path))
    np.save(output_path, x)


def write_pickle(
    x,
    name,
    suffix=None,
    data_path=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
):
    output_path = get_data_file_path(
        name, ".pickle", suffix, data_path, sub_dir, create_sub_dir, verbose
    )
    if verbose:
        log.info("Writing pickle format to %s", data_relative_path(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def is_file_type(target_file_type, file_type=None, suffix=None):
    return (file_type is not None and file_type == target_file_type) or (
        suffix is not None and suffix.endswith(f".{target_file_type}")
    )


def write_data(
    x,
    name,
    suffix=None,
    data_path=None,
    sub_dir=None,
    create_sub_dir=True,
    verbose=False,
    file_type=None,
):
    if is_file_type("csv", file_type, suffix):
        func = write_csv
    elif is_file_type("pkl", file_type, suffix):
        func = write_pickle
    elif is_file_type("npy", file_type, suffix):
        func = write_npy
    else:
        raise ValueError("Could not detect type of file to export to")

    func(
        x=x,
        name=name,
        suffix=suffix,
        data_path=data_path,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
    )


@dataclass
class FileExporter:
    data_path: str = None
    sub_dir: str = None
    suffix: str = None
    create_sub_dir: bool = True
    verbose: bool = False
    file_type: str = "csv"

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def export(self, name, data, suffix=None, sub_dir=None):
        if suffix is None:
            suffix = self.suffix
        if sub_dir is None:
            sub_dir = self.sub_dir
        write_data(
            data,
            name,
            data_path=self.data_path,
            sub_dir=sub_dir,
            suffix=suffix,
            create_sub_dir=self.create_sub_dir,
            verbose=self.verbose,
            file_type=self.file_type,
        )


class NoEmbedExporter:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def export(self, name, embedded):
        pass


@dataclass
class FileEmbedExporter:
    file_exporter: FileExporter

    @classmethod
    def new(cls, **kwargs):
        file_exporter = FileExporter(**kwargs)
        return cls(file_exporter=file_exporter)

    def export(self, name, embedded):
        if isinstance(embedded, dict):
            coords = embedded["coords"]
            self.export_extra(name, embedded)
        else:
            coords = embedded
        self.file_exporter.export(name, coords)

    def export_extra(self, name, embedded, suffix=None):
        if suffix is None:
            suffix = self.file_exporter.suffix
        suffix = ensure_suffix(suffix, self.file_exporter.sub_dir)
        if not islisty(suffix):
            suffix = [suffix]
        for extra_data_name, extra_data in embedded.items():
            if extra_data_name == "coords":
                continue
            if isinstance(extra_data, dict):
                self.export_extra(name, extra_data, suffix=suffix + [extra_data_name])
            else:
                self.file_exporter.export(
                    name, extra_data, suffix=suffix + [extra_data_name]
                )


def create_exporter(embed_method, export=False):
    export, export_kwargs = get_method_and_args(export)
    if isinstance(export, bool):
        if export:
            export = "csv"
        else:
            export = "none"

    if export in ("csv", "pkl", "npy"):
        exporter_cls = FileEmbedExporter
    elif export == "none":
        exporter_cls = NoEmbedExporter
    else:
        raise ValueError(f"Unknown exporter type {export}")

    if export_kwargs is None:
        export_kwargs = dict(suffix=None, create_sub_dir=True, verbose=False)
    if "sub_dir" not in export_kwargs:
        export_kwargs["sub_dir"] = embed_method

    exporter = exporter_cls.new(file_type=export, **export_kwargs)
    return exporter


def create_exporters(embed_method, export=False):
    # bool or string or (file_type, {options}) should be put in a list
    if not islisty(export) or isinstance(export, tuple):
        export = [export]
    return [create_exporter(embed_method, ex) for ex in export]
