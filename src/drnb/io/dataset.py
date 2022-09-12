import pathlib
from dataclasses import dataclass

from drnb.io import DATA_ROOT, FileExporter, get_xy, read_data
from drnb.util import get_method_and_args, get_multi_config


def read_datax(dataset, data_path=None, sub_dir="xy", verbose=False):
    return read_data(
        dataset,
        suffix="",
        data_path=data_path,
        sub_dir=sub_dir,
        verbose=verbose,
        as_numpy=True,
    )


def read_datay(
    dataset,
    data_path=None,
    sub_dir="xy",
    verbose=False,
    target_suffix="y",
    x=None,
    data_suffix="",
):
    try:
        y = read_data(
            dataset,
            suffix=target_suffix,
            data_path=data_path,
            sub_dir=sub_dir,
            verbose=verbose,
        )
    except FileNotFoundError:
        if x is None:
            x = read_data(
                dataset,
                suffix=data_suffix,
                data_path=data_path,
                sub_dir=sub_dir,
                verbose=False,
            )
        y = range(x.shape[0])
    return y


def read_dataxy(
    dataset,
    data_path=None,
    sub_dir="xy",
    verbose=False,
):
    x = read_datax(dataset, data_path=data_path, sub_dir=sub_dir, verbose=verbose)
    y = read_datay(dataset, data_path=data_path, sub_dir=sub_dir, verbose=verbose, x=x)
    return x, y


def read_dataset(
    dataset,
    data_path=None,
    sub_dir="data",
    data_suffix="data",
    target_suffix="target",
    verbose=False,
):
    data = read_data(
        dataset, suffix=data_suffix, data_path=data_path, sub_dir=sub_dir, verbose=False
    )
    target = read_datay(
        dataset,
        data_path=data_path,
        sub_dir=sub_dir,
        verbose=verbose,
        x=data,
        target_suffix=target_suffix,
    )
    return data, target


class XImporter:
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    # pylint: disable=unused-argument
    def import_data(self, name, x, y):
        return get_xy(x, y)


@dataclass
class DatasetImporter:
    data_path: pathlib.Path = DATA_ROOT
    sub_dir: str = "data"
    data_suffix: str = "data"
    target_suffix: str = "target"

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    # pylint: disable=unused-argument
    def import_data(self, name, x=None, y=None):
        data, target = read_dataset(
            name,
            data_path=self.data_path,
            sub_dir=self.sub_dir,
            data_suffix=self.data_suffix,
            target_suffix=self.target_suffix,
        )
        return data, target


def create_dataset_importer(x=None, import_kwargs=None):
    if x is None:
        importer_cls = DatasetImporter
    else:
        importer_cls = XImporter

    if import_kwargs is None:
        import_kwargs = {}

    importer = importer_cls.new(**import_kwargs)
    return importer


def create_dataset_exporter(export_config):
    export, export_kwargs = get_method_and_args(export_config)
    if export in ("csv", "pkl", "npy"):
        exporter_cls = FileExporter
    else:
        raise ValueError(f"Unknown exporter type {export}")

    if export_kwargs is None:
        export_kwargs = dict(suffix=None, create_sub_dir=True, verbose=False)

    exporter = exporter_cls.new(file_type=export, **export_kwargs)
    return exporter


def create_dataset_exporters(export_configs):
    if export_configs is None:
        return None
    export_configs = get_multi_config(export_configs)
    return [create_dataset_exporter(export_config) for export_config in export_configs]
