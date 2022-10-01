from dataclasses import dataclass

from drnb.io import FileExporter, ensure_suffix
from drnb.util import islisty


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


# export=dict(ext=["pkl", "csv"], sub_dir="umap-pl", embed_method_label="densvis")
def create_embed_exporter(
    embed_method_label,
    out_type,
    sub_dir,
    suffix=None,
    create_sub_dir=True,
    drnb_home=None,
    verbose=False,
):
    if suffix is None:
        suffix = embed_method_label
    if not islisty(out_type):
        out_type = [out_type]

    exporters = []
    for otype in out_type:
        if otype in ("csv", "pkl", "npy"):
            exporter_cls = FileEmbedExporter
        else:
            raise ValueError(f"Unknown exporter type {otype}")
        kwargs = dict(
            file_type=otype,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            suffix=suffix,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
        )
        exporters.append(exporter_cls.new(**kwargs))
    return exporters
