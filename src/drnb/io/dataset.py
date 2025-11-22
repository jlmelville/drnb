from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from drnb.io import (
    FileExporter,
    get_drnb_home,
    log,
    read_data,
    read_json,
    read_pickle,
)
from drnb.types import ActionConfig, DataSet
from drnb.util import (
    get_method_and_args,
)


def read_target(
    dataset: str,
    drnb_home: Path | str | None = None,
    sub_dir: str = "data",
    verbose: bool = False,
    target_suffix: str = "target",
    data: np.ndarray | pd.DataFrame = None,
    data_suffix: str = "",
) -> pd.DataFrame:
    """Read the target data for the given dataset with the specified `target_suffix`.
    If not found, the target is assumed to be the range of the number of items in the
    data. If the data is not provided, it is read from the data repository using the
    specified `data_suffix`.
    """
    try:
        target = read_data(
            dataset,
            suffix=target_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            verbose=verbose,
        )
        return target
    except FileNotFoundError:
        if data is None:
            data = read_data(
                dataset,
                suffix=data_suffix,
                drnb_home=drnb_home,
                sub_dir=sub_dir,
                verbose=False,
            )
        target = list(range(data.shape[0]))
        target = pd.DataFrame(target, columns=["target"])
        return target


def read_dataset(
    dataset: str,
    drnb_home: Path | str | None = None,
    sub_dir: str = "data",
    data_suffix: str = "data",
    target_suffix: str = "target",
    verbose: bool = False,
) -> DataSet:
    """Read the data and target for the given dataset. The data is read from the data
    repository using the specified `data_suffix`, and the target is read using the
    specified `target_suffix`. If the target is not found, it is assumed to be the
    range of the number of items in the data."""
    data = read_data(
        dataset, suffix=data_suffix, drnb_home=drnb_home, sub_dir=sub_dir, verbose=False
    )
    target = read_target(
        dataset,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        verbose=verbose,
        data=data,
        target_suffix=target_suffix,
    )
    return data, target


def read_palette(
    dataset: str,
    drnb_home: Path | str | None = None,
    sub_dir: str = "data",
    suffix: str = "target-palette",
    verbose: bool = False,
) -> dict:
    """Read the palette for the given dataset."""
    return read_pickle(
        dataset,
        suffix=suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        verbose=verbose,
    )


@dataclass
class DatasetReader:
    """Class to read datasets from the data repository."""

    drnb_home: Path = get_drnb_home()
    sub_dir: str = "data"
    data_suffix: str = "data"
    target_suffix: str = "target"

    @classmethod
    def new(cls, **kwargs):
        """Create a new object from the given keyword arguments."""
        return cls(**kwargs)

    def read_data(self, name: str) -> DataSet:
        """Read the data and target for the given dataset name."""
        data, target = read_dataset(
            name,
            drnb_home=self.drnb_home,
            sub_dir=self.sub_dir,
            data_suffix=self.data_suffix,
            target_suffix=self.target_suffix,
        )
        return data, target


def create_dataset_exporter(export_config: ActionConfig) -> FileExporter:
    """Create a dataset exporter from the given configuration. The configuration is
    either a string representing the export method (e.g. "csv") or a tuple with the
    first element being the export method and the second element being keyword
    arguments to controlt he export (e.g. suffix)."""
    export, export_kwargs = get_method_and_args(
        export_config, {"suffix": None, "create_sub_dir": True, "verbose": False}
    )
    if export in ("csv", "pkl", "npy", "parquet", "feather"):
        exporter_cls = FileExporter
    else:
        raise ValueError(f"Unknown exporter type {export}")

    exporter = exporter_cls.new(file_type=export, **export_kwargs)
    return exporter


def create_dataset_exporters(
    export_configs: List[ActionConfig] | ActionConfig | None,
) -> List[FileExporter]:
    """Create a list of dataset exporters from the given configuration. The
    configuration is either a list of export configurations, a single export
    configuration, or None. If None, an empty list is returned."""
    if export_configs is None:
        return []
    if not isinstance(export_configs, (list, tuple)):
        export_configs = [export_configs]
    return [create_dataset_exporter(export_config) for export_config in export_configs]


def read_data_pipeline(
    name: str, drnb_home: Path | str | None = None, sub_dir: str | None = "data"
) -> dict:
    """Read the data pipeline for the given dataset name. The pipeline is a JSON file
    with the suffix "pipeline", and is located in the specified data repository."""
    return read_json(name=name, suffix="pipeline", drnb_home=drnb_home, sub_dir=sub_dir)


def filter_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out datasets with bad data, i.e., datasets with missing values for the
    number of items."""
    bad = df["n_items"].isna()
    bad_df = df[bad]
    if not bad_df.empty:
        bad_names = list(bad_df.index)
        log.warning("Bad datasets found: %s", bad_names)
        df = df[~bad]
    return df


def get_dataset_info(
    names: str | List[str],
    drnb_home: Path | str | None = None,
    sub_dir: str | None = "data",
) -> pd.DataFrame:
    """Get the dataset information for the given dataset names in the data repository
    specified by drnb_home and the optional sub_dir (default is "data"). The data
    is returned as a DataFrame with the following columns:

    - name: the dataset name
    - n_items: the number of items in the dataset
    - n_dim: the number of dimensions in the dataset
    - n_target_cols: the number of target columns in the dataset
    - n_na_rows: the number of rows with missing values
    - scale: the scaling method used
    - dim_red: the dimensionality reduction method used
    - n_duplicates: the number of duplicate rows
    - tags: the tags associated with the dataset
    - created_on: the creation date of the dataset
    - updated_on: the last update date of the dataset
    - url: the URL of the dataset
    """
    if not isinstance(names, (list, tuple)):
        return _get_dataset_info(names, drnb_home=drnb_home, sub_dir=sub_dir)

    df = pd.concat(
        [_get_dataset_info(n, drnb_home=drnb_home, sub_dir=sub_dir) for n in names]
    )

    return filter_bad_data(df)


def _get_dataset_info(
    name: str, drnb_home: Path | str | None = None, sub_dir: str | None = "data"
) -> pd.DataFrame:
    """Get the dataset information for the given dataset name. The information is
    extracted from the data pipeline for the dataset. If the pipeline is not found, the
    default information is returned."""

    default_info = {
        "name": name,
        "n_items": None,
        "n_dim": None,
        "n_target_cols": None,
        "n_na_rows": None,
        "scale": None,
        "dim_red": None,
        "n_duplicates": None,
        "tags": None,
        "created_on": None,
        "updated_on": None,
        "url": None,
    }

    try:
        pipeline = read_data_pipeline(name=name, drnb_home=drnb_home, sub_dir=sub_dir)
        pipeline_info = pipeline["pipeline"]

        dshape = pipeline.get("data_shape", (0, 0))
        tshape = pipeline.get("target_shape", (0,))
        n_target_cols = (
            1 if tshape and len(tshape) == 1 else tshape[1] if len(tshape) > 1 else None
        )

        info = {
            "n_items": dshape[0],
            "n_dim": dshape[1],
            "n_target_cols": n_target_cols,
            "n_na_rows": pipeline.get("n_na_rows", 0),
            "scale": pipeline_info.get("scale", {}).get("scale_type", "unknown"),
            "dim_red": pipeline.get("reduce_result", pipeline_info.get("reduce", None)),
            "n_duplicates": pipeline.get("n_duplicates", 0),
            "tags": " ".join(pipeline.get("tags", [])),
            "created_on": pipeline.get("created_on", "unknown"),
            "updated_on": pipeline.get("updated_on", "unknown"),
            "url": pipeline.get("url", ""),
        }

        default_info.update(info)
    except FileNotFoundError:
        pass

    return pd.DataFrame([default_info]).set_index("name")


def list_available_datasets(
    drnb_home: Path | None = None,
    sub_dir: str | None = "data",
    with_target: bool = False,
) -> list[str]:
    """List the available datasets in the data repository specified by drnb_home and
    the optional sub_dir. If sub_dir is None, the default is "data". If with_target is
    True, only datasets with a corresponding target file are returned."""

    if drnb_home is None:
        drnb_home = get_drnb_home()
    data_path = drnb_home
    if sub_dir is not None:
        data_path = drnb_home / sub_dir

    data_suffix = "-data"
    datasets = {
        x.stem[: -len(data_suffix)]
        for x in Path.glob(data_path, "*")
        if x.stem.endswith(data_suffix)
    }

    if with_target:
        target_suffix = "-target"
        # chop off the "-target" bit
        target_stems = {
            t.stem[: -len(target_suffix)]
            for t in Path.glob(data_path, "*")
            if t.stem.endswith(target_suffix)
        }
        # only keep items in datasets and t_stems
        datasets &= target_stems

    return sorted(datasets)


def get_available_dataset_info(
    drnb_home: Path | None = None, sub_dir: str | None = "data"
) -> pd.DataFrame:
    """Get the dataset information for the available datasets in the data repository,
    specified by drnb_home and the optional sub_dir. The data is returned as a DataFrame
    with the following columns:

    - name: the dataset name
    - n_items: the number of items in the dataset
    - n_dim: the number of dimensions in the dataset
    - n_target_cols: the number of target columns in the dataset
    - n_na_rows: the number of rows with missing values
    - scale: the scaling method used
    - dim_red: the dimensionality reduction method used
    - n_duplicates: the number of duplicate rows
    - tags: the tags associated with the dataset
    - created_on: the creation date of the dataset
    - updated_on: the last update date of the dataset
    - url: the URL of the dataset
    """
    names = list_available_datasets(
        drnb_home=drnb_home, sub_dir=sub_dir, with_target=False
    )
    dfs = [get_dataset_info(name) for name in names]
    return filter_bad_data(pd.concat(dfs))
