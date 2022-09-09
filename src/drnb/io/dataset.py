from dataclasses import dataclass

from drnb.io import get_xy, read_data


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


def create_dataset_importer(x=None, import_kwargs=None):
    if x is None:
        importer_cls = DatasetImporter
    else:
        importer_cls = XImporter

    if import_kwargs is None:
        import_kwargs = {}

    importer = importer_cls.new(**import_kwargs)
    return importer
