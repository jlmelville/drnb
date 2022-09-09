from dataclasses import dataclass

from drnb.io import FileExporter, ensure_suffix
from drnb.util import get_method_and_args, islisty


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


def create_embed_exporter(embed_method, export=False):
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


def create_embed_exporters(embed_method, export=False):
    # bool or string or (file_type, {options}) should be put in a list
    if not islisty(export) or isinstance(export, tuple):
        export = [export]
    return [create_embed_exporter(embed_method, ex) for ex in export]
