from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.decomposition

from drnb.io import stringify_paths, write_json, write_pickle
from drnb.io.dataset import create_dataset_exporters
from drnb.log import log, log_verbosity
from drnb.neighbors import NeighborsRequest, create_neighbors_request
from drnb.preprocess import create_scale_kwargs, filter_columns, numpyfy, scale_data
from drnb.triplets import TripletsRequest, create_triplets_request
from drnb.util import Jsonizable, dts_now


@dataclass
class DatasetPipeline(Jsonizable):
    convert: dict = field(default_factory=lambda: dict(dtype="float32", layout="c"))
    scale: dict = field(default_factory=dict)
    check_for_duplicates: bool = False
    reduce: int = None
    reduce_result: Any = None
    drnb_home: Path = None
    data_sub_dir: str = "data"
    data_exporters: list = field(default_factory=list)
    target_exporters: list = field(default_factory=list)
    neighbors_request: NeighborsRequest = None
    triplets_request: TripletsRequest = None
    verbose: bool = False

    def run(
        self,
        name,
        data,
        data_cols=None,
        target=None,
        target_cols=None,
        target_palette=None,
        url=None,
        tags=None,
        verbose=False,
    ):
        with log_verbosity(verbose):
            return self._run(
                name,
                data,
                data_cols=data_cols,
                target=target,
                target_cols=target_cols,
                target_palette=target_palette,
                url=url,
                tags=tags,
            )

    def _run(
        self,
        name,
        data,
        data_cols,
        target,
        target_cols,
        target_palette,
        url,
        tags,
    ):
        if tags is None:
            tags = []

        started_on = dts_now()

        data, target = self.get_target(data, target, target_cols)

        log.info("Initial data shape: %s", data.shape)

        data = self.filter_data_columns(data, data_cols)

        data, dropna_index, n_na_rows = self.dropna(data)

        # as this is potentially memory intensive, it won't be done by default
        # a value of None means "don't know", not "zero duplicates"
        n_duplicates = self.duplicate_check(data)

        data = scale_data(data, **self.scale)

        data = self.convert_data(data)

        (data, reduce_result) = self.reduce_dim(data)

        data_output_paths = self.export_data(data, name)

        target_shape, target_output_paths = self.process_target(
            target,
            name,
            dropna_index,
            target_cols=target_cols,
            target_palette=target_palette,
        )

        neighbors_output_paths = self.calculate_neighbors(data, name)
        triplets_output_paths = self.calculate_triplets(data, name)
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
            triplets_output_paths=triplets_output_paths,
            tags=tags,
            url=url,
            n_na_rows=n_na_rows,
            n_duplicates=n_duplicates,
            reduce_result=reduce_result,
            data_cols=data_cols,
            target_cols=target_cols,
        )
        log.info("Writing pipeline result for %s", name)
        write_json(result, name=name, sub_dir=self.data_sub_dir, suffix="pipeline")

        return result

    def dropna(self, data):
        nrows_before = data.shape[0]
        log.info("Removing rows with NAs")
        if isinstance(data, pd.DataFrame):
            data_nona = data.dropna()
            data = data_nona
            data_nona_index = data_nona.index
        else:
            data_nona_index = ~np.isnan(data).any(axis=1)
            data = data[data_nona_index]

        nrows_after = data.shape[0]
        n_na_rows = nrows_before - nrows_after
        log.info("Data shape after filtering NAs: %s", data.shape)
        return data, data_nona_index, n_na_rows

    def filter_data_columns(self, data, data_cols):
        data = filter_columns(data, data_cols)
        log.info("Data shape after filtering columns: %s", data.shape)
        return data

    def duplicate_check(self, data):
        if not self.check_for_duplicates:
            return None
        n_duplicates = data.shape[0] - np.unique(data, axis=0).shape[0]
        log.info("Checked for duplicates: found %d", n_duplicates)
        return n_duplicates

    def convert_data(self, data):
        if self.convert is not None:
            log.info("Converting to numpy with %s", self.convert)
            data = numpyfy(data, **self.convert)
        return data

    def reduce_dim(self, data):
        if self.reduce is None:
            return data, None
        log.info("Reducing initial dimensionality to %d", self.reduce)
        pca = sklearn.decomposition.PCA(n_components=self.reduce).fit(data)
        varex = float(np.sum(pca.explained_variance_ratio_) * 100.0)
        log.info(
            "PCA: %d components explain %.2f%% of variance",
            self.reduce,
            varex,
        )
        data = pca.transform(data)
        log.info("Data shape after PCA: %s", data.shape)
        reduce_result = f"PCA {self.reduce} ({varex:.2f}%)"
        return data, reduce_result

    def get_target(self, data, target, target_cols=None):
        if target is not None or (target_cols is not None and target_cols):
            if target is None:
                log.info("Using data as source for target")
                target = data
        return data, target

    def process_target(
        self, target, name, dropna_index, target_cols=None, target_palette=None
    ):
        if isinstance(target, pd.Series):
            target = target.to_frame()
        target_shape = None
        target_output_paths = []
        if target is not None:
            log.info("Processing target with initial shape %s", target.shape)
            target = target.loc[dropna_index]
            target = filter_columns(target, target_cols)
            target_shape = target.shape
            if self.target_exporters is None or not self.target_exporters:
                log.warning("Target supplied but no target exporters defined")
            else:
                target_output_paths = self.export_target(target, name)
            if target_palette:
                target_palette_path = write_pickle(
                    target_palette,
                    name,
                    suffix="target-palette",
                    drnb_home=None,
                    sub_dir=self.data_sub_dir,
                    create_sub_dir=True,
                    verbose=True,
                )
                target_output_paths.append(stringify_paths([target_palette_path]))
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
                name,
                data,
                drnb_home=self.drnb_home,
                sub_dir=self.data_sub_dir,
                suffix=what,
            )
            all_output_paths += stringify_paths(output_paths)
        return all_output_paths

    def calculate_neighbors(self, data, name):
        if self.neighbors_request is None:
            return None
        log.info("Calculating nearest neighbors")

        neighbors_output_paths = self.neighbors_request.create_neighbors(
            data, dataset_name=name, nbr_dir="nn"
        )

        return stringify_paths(neighbors_output_paths)

    def calculate_triplets(self, data, name):
        if self.triplets_request is None:
            return None
        log.info("Calculating triplets")

        triplet_output_paths = self.triplets_request.create_triplets(
            data, dataset_name=name, triplet_dir="triplets"
        )

        return stringify_paths(triplet_output_paths)


@dataclass
class DatasetPipelineResult(Jsonizable):
    pipeline: str
    started_on: str = "unknown"
    created_on: str = "unknown"
    updated_on: str = "unknown"
    data_shape: tuple = None
    n_na_rows: int = 0
    n_duplicates: int = None
    reduce_result: str = None
    data_output_paths: list = field(default_factory=list)
    target_shape: tuple = None
    target_output_paths: list = field(default_factory=list)
    neighbors_output_paths: list = field(default_factory=list)
    triplets_output_paths: list = field(default_factory=list)
    url: str = None
    tags: list = field(default_factory=list)
    data_cols: list = field(default_factory=list)
    target_cols: list = field(default_factory=list)


def create_data_pipeline(
    data_export,
    check_for_duplicates=False,
    convert=True,
    scale=None,
    reduce=None,
    target_export=None,
    neighbors=None,
    triplets=None,
    drnb_home=None,
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
            drnb_home=drnb_home,
            check_for_duplicates=check_for_duplicates,
            convert=convert,
            scale=create_scale_kwargs(scale),
            reduce=reduce,
            data_exporters=data_exporters,
            target_exporters=target_exporters,
            neighbors_request=create_neighbors_request(neighbors),
            triplets_request=create_triplets_request(triplets),
            verbose=verbose,
        )


def create_default_pipeline(
    drnb_home=None,
    check_for_duplicates=False,
    scale=None,
    reduce=None,
    csv=True,
    metric=None,
):
    if metric is None:
        metric = ["euclidean"]
    data_export = ["npy"]
    target_export = ["pkl"]
    triplet_file_types = ["npy"]
    neighbor_file_types = ["npy"]
    if csv:
        data_export.append("csv")
        target_export.append("csv")
        triplet_file_types.append("csv")
        neighbor_file_types.append("csv")
    return create_data_pipeline(
        drnb_home=drnb_home,
        check_for_duplicates=check_for_duplicates,
        scale=scale,
        reduce=reduce,
        data_export=data_export,
        target_export=target_export,
        neighbors=dict(
            n_neighbors=[15, 50, 150],
            method="exact",
            metric=metric,
            file_types=triplet_file_types,
        ),
        triplets=dict(
            n_triplets_per_point=5,
            seed=1337,
            file_types=neighbor_file_types,
        ),
        verbose=True,
    )
