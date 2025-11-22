from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from drnb.io import FileExporter, ensure_suffix


class NoEmbedExporter:
    """Do nothing with the embedded data when not exporting"""

    def __init__(self, **_):
        pass

    @classmethod
    def new(cls, **kwargs):
        """Create a new NoEmbedExporter."""
        return cls(**kwargs)

    def export(self, _, __):
        """Do nothing."""


@dataclass
class FileEmbedExporter:
    """Export embedded data to a file."""

    file_exporter: FileExporter

    @classmethod
    def new(cls, **kwargs):
        """Create a new FileEmbedExporter."""
        file_exporter = FileExporter(**kwargs)
        return cls(file_exporter=file_exporter)

    def export(self, name: str, embedded: dict | np.ndarray):
        """Export the embedded data."""
        if isinstance(embedded, dict):
            coords = embedded["coords"]
            self.export_extra(name, embedded)
        else:
            coords = embedded
        self.file_exporter.export(name, coords)

    def export_extra(
        self, name: str, embedded: dict, suffix: str | List[str] | None = None
    ):
        """Export extra data from the `embedded` dict."""
        if suffix is None:
            suffix = self.file_exporter.suffix
        suffix = ensure_suffix(suffix, self.file_exporter.sub_dir)
        if not isinstance(suffix, (list, tuple)):
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
    embed_method_label: str,
    out_type: str | List[str],
    sub_dir: str | None = None,
    suffix: str | List[str] | None = None,
    create_sub_dir: bool = True,
    drnb_home: Path | str | None = None,
    verbose: bool = False,
) -> List[FileEmbedExporter]:
    """Create exporters for embedded data."""

    if suffix is None:
        suffix = embed_method_label
    if not isinstance(out_type, (list, tuple)):
        out_type = [out_type]

    exporters = []
    for otype in out_type:
        if otype in ("csv", "pkl", "npy"):
            exporter_cls = FileEmbedExporter
        else:
            raise ValueError(f"Unknown exporter type {otype}")
        kwargs = {
            "file_type": otype,
            "drnb_home": drnb_home,
            "sub_dir": sub_dir,
            "suffix": suffix,
            "create_sub_dir": create_sub_dir,
            "verbose": verbose,
        }
        exporters.append(exporter_cls.new(**kwargs))
    return exporters
