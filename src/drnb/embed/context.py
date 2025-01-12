from dataclasses import dataclass
from pathlib import Path

import numpy as np

import drnb.io as nbio
import drnb.io.dataset as dataio
from drnb.neighbors import (
    NearestNeighbors,
    get_neighbors,
)
from drnb.types import DataSet


@dataclass
# pylint: disable=too-many-instance-attributes
class EmbedContext:
    """Context for embedding algorithms. Used to store information about where to
    find data used by the embedding algorithms and where to save the results.

    Attributes:
        dataset_name: Name of the dataset.
        embed_method_name: Name of the embedding method.
        embed_method_variant: Variant of the embedding method.
        drnb_home: Path to the root directory of the DRNB project.
        data_sub_dir: Subdirectory where the data is stored.
        nn_sub_dir: Subdirectory where the nearest neighbors are stored.
        triplet_sub_dir: Subdirectory where the triplets are stored.
        experiment_name: Name of the experiment.
    """

    dataset_name: str
    embed_method_name: str
    embed_method_variant: str = ""
    drnb_home: Path | None = nbio.get_drnb_home_maybe()
    data_sub_dir: str = "data"
    nn_sub_dir: str = "nn"
    triplet_sub_dir: str = "triplets"
    experiment_name: str | None = None

    @property
    def embed_method_label(self) -> str:
        """Get the label for the embedding method."""
        if self.embed_method_variant:
            return self.embed_method_variant
        return self.embed_method_name

    @property
    def embed_nn_name(self) -> str:
        """Get the name of the nearest neighbors."""
        return f"{self.dataset_name}-{self.embed_method_label}-nn"

    @property
    def embed_triplets_name(self) -> str:
        """Get the name of the triplets."""
        return f"{self.dataset_name}-{self.embed_method_label}-triplets"


def get_neighbors_with_ctx(
    data: np.ndarray,
    metric: str,
    n_neighbors: int,
    knn_params: dict | None = None,
    ctx: EmbedContext | None = None,
    return_distance: bool = True,
) -> NearestNeighbors:
    """Get nearest neighbors using the provided context."""
    if knn_params is None:
        knn_params = {}
    knn_defaults = {"method": "exact", "cache": False, "verbose": True, "name": None}
    if ctx is not None:
        knn_defaults.update(
            {
                "drnb_home": ctx.drnb_home,
                "sub_dir": ctx.nn_sub_dir,
                "name": ctx.dataset_name,
            }
        )
    full_knn_params = knn_defaults | knn_params

    # only turn on caching request if there is an actual name in the context
    name = knn_defaults.get("name")
    knn_defaults["cache"] = name is not None and name

    result = get_neighbors(
        data=data,
        n_neighbors=n_neighbors,
        metric=metric,
        return_distance=return_distance,
        **full_knn_params,
    )
    # let's just make sure we get the dist member
    if return_distance and result.dist is None:
        raise ValueError("return_distance was True but no distance data was returned")
    return result


def read_dataset_from_ctx(ctx: EmbedContext, verbose: bool = False) -> DataSet:
    """Read the dataset from the context."""
    return dataio.read_dataset(
        dataset=ctx.dataset_name,
        drnb_home=ctx.drnb_home,
        sub_dir=ctx.data_sub_dir,
        verbose=verbose,
    )
