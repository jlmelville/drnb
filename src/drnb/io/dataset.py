from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drnb.io import FileExporter, get_drnb_home, islisty, read_data, read_json
from drnb.util import (
    READABLE_DATETIME_FMT,
    dts_to_str,
    get_method_and_args,
    get_multi_config,
)


def read_target(
    dataset,
    drnb_home=None,
    sub_dir="data",
    verbose=False,
    target_suffix="target",
    data=None,
    data_suffix="",
):
    try:
        return read_data(
            dataset,
            suffix=target_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            verbose=verbose,
        )
    except FileNotFoundError:
        if data is None:
            data = read_data(
                dataset,
                suffix=data_suffix,
                drnb_home=drnb_home,
                sub_dir=sub_dir,
                verbose=False,
            )
        target = range(data.shape[0])
        return target


def read_dataset(
    dataset,
    drnb_home=None,
    sub_dir="data",
    data_suffix="data",
    target_suffix="target",
    verbose=False,
):
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


@dataclass
class DatasetImporter:
    drnb_home: Path = None
    sub_dir: str = "data"
    data_suffix: str = "data"
    target_suffix: str = "target"

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    # pylint: disable=unused-argument
    def import_data(self, name, x=None, y=None):
        drnb_home = self.drnb_home
        if drnb_home is None:
            drnb_home = get_drnb_home()
        data, target = read_dataset(
            name,
            drnb_home=drnb_home,
            sub_dir=self.sub_dir,
            data_suffix=self.data_suffix,
            target_suffix=self.target_suffix,
        )
        return data, target


def create_dataset_exporter(export_config):
    export, export_kwargs = get_method_and_args(
        export_config, dict(suffix=None, create_sub_dir=True, verbose=False)
    )
    if export in ("csv", "pkl", "npy"):
        exporter_cls = FileExporter
    else:
        raise ValueError(f"Unknown exporter type {export}")

    exporter = exporter_cls.new(file_type=export, **export_kwargs)
    return exporter


def create_dataset_exporters(export_configs):
    if export_configs is None:
        return None
    export_configs = get_multi_config(export_configs)
    return [create_dataset_exporter(export_config) for export_config in export_configs]


def read_data_pipeline(name, drnb_home=None, sub_dir="data"):
    return read_json(name=name, suffix="pipeline", drnb_home=drnb_home, sub_dir=sub_dir)


def get_dataset_info(names, drnb_home=None, sub_dir="data"):
    if not islisty(names):
        return _get_dataset_info(names, drnb_home=drnb_home, sub_dir=sub_dir)
    return pd.concat(
        [_get_dataset_info(n, drnb_home=drnb_home, sub_dir=sub_dir) for n in names]
    )


def _get_dataset_info(name, drnb_home=None, sub_dir="data"):
    pipeline = read_data_pipeline(name=name, drnb_home=drnb_home, sub_dir=sub_dir)
    dshape = pipeline["data_shape"]
    tshape = pipeline["target_shape"]
    pipeline_info = pipeline["pipeline"]
    created_on = pipeline["created_on"]
    updated_on = pipeline["updated_on"]
    scale = pipeline_info["scale"]["scale_type"]
    n_na_rows = pipeline.get("n_na_rows", 0)
    n_duplicates = pipeline.get("n_duplicates")
    tags = pipeline.get("tags", [])
    tags = " ".join(tags)
    url = pipeline.get("url", "")
    dim_red = pipeline.get("reduce_result")
    if dim_red is None:
        dim_red = pipeline_info.get("reduce")
    if tshape is not None:
        if len(tshape) == 1:
            n_target_cols = 1
        else:
            n_target_cols = tshape[1]
    else:
        n_target_cols = None
    return pd.DataFrame(
        dict(
            name=name,
            n_items=dshape[0],
            n_dim=dshape[1],
            n_target_cols=n_target_cols,
            n_na_rows=n_na_rows,
            scale=scale,
            dim_red=dim_red,
            n_duplicates=n_duplicates,
            tags=tags,
            created_on=dts_to_str(created_on, READABLE_DATETIME_FMT),
            updated_on=dts_to_str(updated_on, READABLE_DATETIME_FMT),
            url=url,
        ),
        index=[0],
    ).set_index("name")


def list_available_datasets(drnb_home=None, sub_dir="data", with_target=False):
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


def get_available_dataset_info(drnb_home=None, sub_dir="data", names=None):
    names = list_available_datasets(
        drnb_home=drnb_home, sub_dir=sub_dir, with_target=False
    )
    dfs = [get_dataset_info(name) for name in names]
    return pd.concat(dfs)
