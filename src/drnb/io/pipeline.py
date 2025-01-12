from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import pandas as pd
import sklearn.decomposition

from drnb.io import FileExporter, stringify_paths, write_json, write_pickle
from drnb.io.dataset import create_dataset_exporters
from drnb.log import log, log_verbosity
from drnb.neighbors import NeighborsRequest, create_neighbors_request
from drnb.preprocess import filter_columns, numpyfy, scale_data
from drnb.triplets import TripletsRequest, create_triplets_request
from drnb.types import ActionConfig
from drnb.util import Jsonizable, dts_to_str


@dataclass
# pylint: disable=too-many-instance-attributes
class DatasetPipeline(Jsonizable):
    """Data class to store and run a data pipeline."""

    convert_args: dict | None = field(
        default_factory=lambda: {"dtype": "float32", "layout": "c"}
    )
    scale_action: ActionConfig | None = None
    check_for_duplicates: bool = False
    reduce: int | None = None
    reduce_result: Any = None
    drnb_home: Path | str | None = None
    data_sub_dir: str = "data"
    data_exporters: List[FileExporter] | None = field(default_factory=list)
    target_exporters: List[FileExporter] | None = field(default_factory=list)
    neighbors_request: NeighborsRequest | None = None
    triplets_request: TripletsRequest | None = None
    verbose: bool = False

    def run(
        self,
        name,
        data,
        data_cols=None,
        target=None,
        target_cols: List[str] = None,
        target_palette=None,
        url=None,
        tags=None,
        verbose=False,
    ):
        """Run the data pipeline on the given data. Return the pipeline result."""
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
        name: str,
        data: np.ndarray | pd.DataFrame,
        data_cols: List[int] | List[str] | None,
        target: np.ndarray | pd.DataFrame | pd.Series | None,
        target_cols: List[str] | None,
        target_palette,
        url,
        tags,
    ):
        """Run the data pipeline on the given data. Return the pipeline result."""
        if tags is None:
            tags = []

        started_on = dts_to_str()

        data, target = self.get_target(data, target, target_cols)

        log.info("Initial data shape: %s", data.shape)

        data = self.filter_data_columns(data, data_cols)

        data, dropna_index, n_na_rows = self.dropna(data)

        # as this is potentially memory intensive, it won't be done by default
        # a value of None means "don't know", not "zero duplicates"
        n_duplicates = self.duplicate_check(data)

        data = self.scale_data(data)

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

        created_on = dts_to_str()

        result = DatasetPipelineResult(
            str(self),
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

    def dropna(
        self, data: np.ndarray | pd.DataFrame
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray, int]:
        """Remove rows with NAs from the data. Return the filtered data,
        a boolean mask of the rows that were kept, and the number of rows removed."""
        nrows_before = data.shape[0]
        log.info("Removing rows with NAs")
        if isinstance(data, pd.DataFrame):
            data_nona = data.dropna()
            data_nona_index = data.index.isin(data_nona.index)
            data = data_nona
        else:
            data_nona_index = ~np.isnan(data).any(axis=1)
            data = data[data_nona_index]

        nrows_after = data.shape[0]
        n_na_rows = nrows_before - nrows_after
        log.info("Data shape after filtering NAs: %s", data.shape)
        return data, data_nona_index, n_na_rows

    def filter_data_columns(
        self, data: np.ndarray | pd.DataFrame, data_cols: List[int] | List[str] | None
    ) -> np.ndarray | pd.DataFrame:
        """Filter the columns of the data to keep only the ones in `data_cols`."""
        data = filter_columns(data, data_cols)
        log.info("Data shape after filtering columns: %s", data.shape)
        return data

    def duplicate_check(self, data: np.ndarray | pd.DataFrame) -> int | None:
        """Check for duplicates in the data. Return the number of duplicates found."""
        if not self.check_for_duplicates:
            return None
        n_duplicates = data.shape[0] - np.unique(data, axis=0).shape[0]
        log.info("Checked for duplicates: found %d", n_duplicates)
        return n_duplicates

    def scale_data(self, data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Scale the data using the given method."""
        if self.scale_action is None:
            return data
        data = scale_data(data, self.scale_action)
        return data

    def convert_data(
        self, data: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        """Convert the data to a numpy array."""
        if self.convert_args is not None:
            log.info("Converting to numpy with %s", self.convert_args)
            data = numpyfy(data, **self.convert_args)
        return data

    def reduce_dim(
        self, data: np.ndarray | pd.DataFrame
    ) -> tuple[np.ndarray | pd.DataFrame, str | None]:
        """Reduce the dimensionality of the data using PCA."""
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

    def get_target(
        self,
        data: np.ndarray | pd.DataFrame,
        target: np.ndarray | pd.DataFrame | pd.Series | None,
        target_cols: List[str] | None = None,
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame | pd.Series | None]:
        """Get the target data, which can be the same as the input data."""
        # if we are given target columns but no target data assume we are using the data
        # as the target
        if target_cols and target is None:
            log.info("Using data as source for target info")
            target = data
        return data, target

    def process_target(
        self,
        target: np.ndarray | pd.DataFrame | pd.Series | None,
        name: str,
        dropna_index: np.ndarray,
        target_cols: List[str] | None = None,
        target_palette: dict | None = None,
    ) -> tuple[tuple[int, int] | None, List[str]]:
        """Process the target data, if any. Return the shape of the target data and the
        paths to the exported target files."""
        if isinstance(target, np.ndarray):
            target = pd.DataFrame(target)
        if isinstance(target, pd.Series):
            target = target.to_frame()
        target_shape = None
        target_output_paths = []
        if target is None:
            return target_shape, target_output_paths

        log.info("Processing target with initial shape %s", target.shape)
        target = target.loc[dropna_index]
        target = filter_columns(target, target_cols)

        # feather format does not support integer column names
        target.columns = [str(c) for c in target.columns]
        # and also does not serialize an index, so we need to "promote" it to a column
        target = target.reset_index()

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
                drnb_home=self.drnb_home,
                sub_dir=self.data_sub_dir,
                create_sub_dir=True,
                verbose=True,
            )
            target_output_paths.append(stringify_paths([target_palette_path]))
        return target_shape, target_output_paths

    def export_data(
        self,
        data: np.ndarray | pd.DataFrame,
        name: str,
    ):
        """Export the data using the data exporters. Return the paths to the exported
        files."""
        return self.export(data, name, self.data_exporters, what="data")

    def export_target(
        self,
        data: np.ndarray | pd.DataFrame,
        name: str,
    ):
        """Export the target data using the target exporters. Return the paths to the
        exported files."""
        return self.export(data, name, self.target_exporters, what="target")

    def export(
        self,
        data: np.ndarray | pd.DataFrame,
        name: str,
        exporters: List[FileExporter],
        what: str,
    ) -> List[str]:
        """Export the data using the exporters. Return the paths to the exported
        files."""
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

    def calculate_neighbors(
        self, data: pd.DataFrame | np.ndarray, name: str
    ) -> List[str]:
        """Calculate nearest neighbors for the data. Return the paths to the neighbor
        files."""
        if self.neighbors_request is None:
            return []
        log.info("Calculating nearest neighbors")

        neighbors_output_paths = self.neighbors_request.create_neighbors(
            data, dataset_name=name, nbr_dir="nn"
        )

        return stringify_paths(neighbors_output_paths)

    def calculate_triplets(
        self, data: pd.DataFrame | np.ndarray, name: str
    ) -> List[str]:
        """Calculate triplets for the data. Return the paths to the triplet files."""
        if self.triplets_request is None:
            return []
        log.info("Calculating triplets")

        triplet_output_paths = self.triplets_request.create_triplets(
            data, dataset_name=name, triplet_dir="triplets"
        )

        return stringify_paths(triplet_output_paths)


@dataclass
# pylint: disable=too-many-instance-attributes
class DatasetPipelineResult(Jsonizable):
    """Data class to store the results of a data pipeline run."""

    pipeline: str
    started_on: str = "unknown"
    created_on: str = "unknown"
    updated_on: str = "unknown"
    data_shape: tuple | None = None
    n_na_rows: int = 0
    n_duplicates: int | None = None
    reduce_result: str | None = None
    data_output_paths: list = field(default_factory=list)
    target_shape: tuple | None = None
    target_output_paths: List[str] | None = field(default_factory=list)
    neighbors_output_paths: List[str] | None = field(default_factory=list)
    triplets_output_paths: List[str] | None = field(default_factory=list)
    url: str | None = None
    tags: list = field(default_factory=list)
    data_cols: List[int] | List[str] | None = field(default_factory=list)
    target_cols: list = field(default_factory=list)


def create_data_pipeline(
    data_export: ActionConfig | List[ActionConfig],
    check_for_duplicates: bool = False,
    convert: bool | dict | None = True,
    scale: ActionConfig | None | Literal[False] = None,
    reduce: int | None = None,
    target_export: ActionConfig | List[ActionConfig] | None = None,
    neighbors: dict | None = None,
    triplets: dict | None = None,
    drnb_home: Path | str | None = None,
    verbose: bool = False,
) -> DatasetPipeline:
    """Create a data pipeline with the given parameters."""
    if isinstance(convert, bool):
        if convert:
            convert_args = {"dtype": "float32", "layout": "c"}
        else:
            convert_args = None
    else:
        convert_args = convert

    if scale is False:
        scale = None

    with log_verbosity(verbose):
        data_exporters = create_dataset_exporters(data_export)
        target_exporters = create_dataset_exporters(target_export)

        return DatasetPipeline(
            drnb_home=drnb_home,
            check_for_duplicates=check_for_duplicates,
            convert_args=convert_args,
            scale_action=scale,
            reduce=reduce,
            data_exporters=data_exporters,
            target_exporters=target_exporters,
            neighbors_request=create_neighbors_request(neighbors),
            triplets_request=create_triplets_request(triplets),
            verbose=verbose,
        )


def create_default_pipeline(
    drnb_home: Path | str | None = None,
    check_for_duplicates: bool = False,
    scale: ActionConfig | None | Literal[False] = None,
    reduce: int | None = None,
    csv: bool = True,
    metric: str | List[str] | None = None,
    verbose: bool = True,
) -> DatasetPipeline:
    """Create a default data pipeline. Some limited overriding of defaults is available
    via the given parameters."""
    if metric is None:
        metric = ["euclidean"]

    data_export = ["npy"]
    target_export = ["feather"]
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
        neighbors={
            "n_neighbors": [15, 50, 150],
            "method": "exact",
            "metric": metric,
            "file_types": neighbor_file_types,
        },
        triplets={
            "n_triplets_per_point": 5,
            "seed": 1337,
            "file_types": triplet_file_types,
        },
        verbose=verbose,
    )
