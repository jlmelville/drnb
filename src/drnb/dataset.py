from dataclasses import dataclass, field

import numpy as np

from drnb.io import data_relative_path, write_json
from drnb.io.dataset import create_dataset_exporters
from drnb.log import log, log_verbosity
from drnb.neighbors import (
    NeighborsRequest,
    calculate_neighbors,
    slice_neighbors,
    write_neighbors,
)
from drnb.preprocess import create_scale_kwargs, filter_columns, numpyfy, scale_data
from drnb.util import Jsonizable, dts_now, islisty


@dataclass
class DatasetPipeline(Jsonizable):
    data_cols: list = field(default_factory=list)
    convert: dict = field(default_factory=lambda: dict(dtype="float32", layout="c"))
    scale: dict = field(default_factory=dict)
    data_sub_dir: str = "data"
    data_exporters: list = field(default_factory=list)
    target_cols: list = field(default_factory=list)
    target_exporters: list = field(default_factory=list)
    neighbors_request: NeighborsRequest = None

    verbose: bool = False

    def run(self, name, data, target=None, verbose=False):
        with log_verbosity(verbose):
            return self.runv(name, data, target=target)

    def runv(self, name, data, target):
        started_on = dts_now()

        data, target = self.get_target(data, target)

        log.info("initial data shape: %s target shape: %s", data.shape, target.shape)

        data, dropna_index = self.dropna(data)

        data = self.filter_data_columns(data)

        data = scale_data(data, **self.scale)

        data = self.convert_data(data)

        data_output_paths = self.export_data(data, name)

        target_shape, target_output_paths = self.process_target(
            target, dropna_index, name
        )

        neighbors_output_paths = self.calculate_neighbors(data, name)

        created_on = dts_now()
        result = DatasetPipelineResult(
            self,
            data_shape=data.shape,
            data_output_paths=data_output_paths,
            target_shape=target_shape,
            target_output_paths=target_output_paths,
            started_on=started_on,
            created_on=created_on,
            updated_on=created_on,
            neighbors_output_paths=neighbors_output_paths,
        )
        log.info("Writing pipeline result for %s", name)
        write_json(result, name=name, sub_dir=self.data_sub_dir, suffix="pipeline")

        return result

    def dropna(self, data):
        log.info("Removing rows with NAs")
        data_nona = data.dropna()
        data = data_nona
        log.info("data shape after filtering NAs: %s", data.shape)
        return data, data_nona.index

    def filter_data_columns(self, data):
        data = filter_columns(data, self.data_cols)
        log.info("data shape after filtering columns: %s", data.shape)
        return data

    def convert_data(self, data):
        if self.convert is not None:
            log.info("Converting to numpy with %s", self.convert)
            data = numpyfy(data, **self.convert)
        return data

    def get_target(self, data, target):
        if target is not None or (self.target_cols is not None and self.target_cols):
            if target is None:
                log.info("Using data as source for target")
                target = data
        return data, target

    def process_target(self, target, dropna_index, name):
        target_shape = None
        target_output_paths = []
        if target is not None:
            log.info("Processing target")
            target = target.loc[dropna_index]
            target = filter_columns(target, self.target_cols)
            target_shape = target.shape
            if self.target_exporters is None or not self.target_exporters:
                log.warning("Target supplied but no target exporters defined")
            else:
                target_output_paths = self.export_target(target, name)
        return target_shape, target_output_paths

    def export_data(self, data, name):
        return self.export(data, name, self.data_exporters, what="data")

    def export_target(self, data, name):
        return self.export(data, name, self.target_exporters, what="target")

    def export(self, data, name, exporters, what):
        all_output_paths = []
        log.info("Writing %s for %s", what, name)
        for exporter in exporters:
            output_paths = exporter.export(
                name, data, sub_dir=self.data_sub_dir, suffix=what
            )
            all_output_paths += stringify_paths(output_paths)
        return all_output_paths

    def calculate_neighbors(self, data, name):
        neighbors_output_paths = None
        if self.neighbors_request is not None:
            log.info("Calculating nearest neighbors")
            for metric in self.neighbors_request.metric:
                max_n_neighbors = np.max(self.neighbors_request.n_neighbors)
                neighbors_data = calculate_neighbors(
                    data=data,
                    n_neighbors=max_n_neighbors,
                    metric=metric,
                    return_distance=True,
                    **self.neighbors_request.params,
                    verbose=self.verbose,
                    name=name
                )
                neighbors_output_paths = []
                for n_neighbors in self.neighbors_request.n_neighbors:
                    sliced_neighbors = slice_neighbors(neighbors_data, n_neighbors)
                    idx_paths, dist_paths = write_neighbors(
                        neighbor_data=sliced_neighbors,
                        sub_dir="nn",
                        create_sub_dir=True,
                        file_type=self.neighbors_request.file_types,
                        verbose=False,
                    )
                    neighbors_output_paths.append(
                        stringify_paths(idx_paths + dist_paths)
                    )
        return neighbors_output_paths


def stringify_paths(paths):
    return [str(data_relative_path(path)) for path in paths]


@dataclass
class DatasetPipelineResult(Jsonizable):
    pipeline: str
    started_on: str = "unknown"
    created_on: str = "unknown"
    updated_on: str = "unknown"
    data_shape: tuple = None
    data_output_paths: list = field(default_factory=list)
    target_shape: tuple = None
    target_output_paths: list = field(default_factory=list)
    neighbors_output_paths: list = field(default_factory=list)


def create_data_pipeline(
    data_export,
    data_cols=None,
    convert=True,
    scale=None,
    target_cols=None,
    target_export=None,
    neighbors=None,
    verbose=False,
):
    if isinstance(convert, bool):
        if convert:
            convert = dict(dtype="float32", layout="c")
        else:
            convert = None

    with log_verbosity(verbose):
        data_exporters = create_dataset_exporters(data_export)
        target_exporters = create_dataset_exporters(target_export)

        return DatasetPipeline(
            convert=convert,
            scale=create_scale_kwargs(scale),
            data_cols=data_cols,
            target_cols=target_cols,
            data_exporters=data_exporters,
            target_exporters=target_exporters,
            neighbors_request=create_neighbors_request(neighbors),
            verbose=verbose,
        )


#   # for method = "exact" or "approximate" we can't know what algo we will get
#   # so need to nest the names inside?
#   method_kwds = dict("annoy"=dict(), hnsw=dict() ... )
def create_neighbors_request(neighbors_kwds):
    if neighbors_kwds is None:
        return None
    for key in ["metric", "n_neighbors"]:
        if key in neighbors_kwds and not islisty(neighbors_kwds[key]):
            neighbors_kwds[key] = [neighbors_kwds[key]]
    neighbors_request = NeighborsRequest.new(**neighbors_kwds)
    log.info("Requesting one extra neighbor to account for self-neighbor")
    neighbors_request.n_neighbors = [
        n_nbrs + 1 for n_nbrs in neighbors_request.n_neighbors
    ]
    return neighbors_request
