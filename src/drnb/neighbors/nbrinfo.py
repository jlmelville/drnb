from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from drnb.log import log


def idx_to_dist(idx_path: Path) -> Path:
    """Given the path to an index file, return the path to the corresponding distances
    file."""
    dist_stem = idx_path.stem[:-3] + "dist"
    dist_path = idx_path.parent / (dist_stem + idx_path.suffix)
    return dist_path


# everything between name and the extension (exclusive)
# e.g. iris.15.euclidean.exact.faiss.idx.npy => .15.euclidean.exact.faiss.idx
def nn_suffix(path: Path) -> str:
    """Return the suffix of a nearest neighbors file (excluding the name and the
    extension).

    e.g. iris.15.euclidean.exact.faiss.idx.npy => .15.euclidean.exact.faiss.idx
    """
    return "".join(path.suffixes[:-1])


@dataclass
# pylint:disable=too-many-instance-attributes
class NbrInfo:
    """Information about the nearest neighbors of a dataset.

    Fields:

    - name: str: name of the dataset
    - n_nbrs: int: number of neighbors
    - metric: str: e.g. "euclidean"
    - exact: bool: exact or approximate neighbors
    - method: str: method to generate e.g. "faiss", "annoy", "hnsw"
    - has_distances: bool: True if the distances are present
    - idx_path: Path | None: the path to the idx file (optional)
    - dist_path: Path | None: the path to the dist file (if it exists)
    """

    name: str  # name of the dataset
    n_nbrs: int  # number of neighbors
    metric: str  # e.g. "euclidean"
    exact: bool  # exact or approximate neighbors
    method: str  # method to generate e.g. "faiss", "annoy", "hnsw"
    has_distances: bool  # True if the distances are present
    idx_path: Path | None  # the path to the idx file (optional)
    dist_path: Path | None  # the path to the dist file (if it exists)

    @property
    def idx_suffix(self) -> str:
        """Return the suffix of the index file."""
        if self.idx_path is not None:
            return nn_suffix(self.idx_path)
        return self._suffix("idx")

    @property
    def dist_suffix(self) -> str:
        """Return the suffix of the distances file."""
        if self.dist_path is not None:
            return nn_suffix(self.dist_path)
        return self._suffix("dist")

    def _suffix(self, nn_component: str):
        return "." + ".".join(
            [
                str(self.n_nbrs),
                self.metric,
                self.neighbor_type,
                self.method,
                nn_component,
            ]
        )

    @property
    def neighbor_type(self) -> str:
        """Return the type of neighbors (exact or approximate)."""
        if self.exact:
            return "exact"
        return "approximate"

    @classmethod
    def from_path(cls, idx_path: Path, ignore_bad_path: bool = False) -> Self | None:
        """Create a NbrInfo object from the path to an index file."""
        items = idx_path.stem.split(".")
        if len(items) != 6:
            msg = f"Unknown nn file format: {idx_path}"
            if ignore_bad_path:
                log.warning(msg)
                return None
            raise ValueError(msg)
        name = items[0]
        n_nbrs = int(items[1])
        metric = items[2]
        if items[3] == "exact":
            exact = True
        elif items[3] == "approximate":
            exact = False
        else:
            msg = f"Unknown nn file format: {idx_path}"
            if ignore_bad_path:
                log.warning(msg)
                return None
            raise ValueError(msg)
        method = items[4]

        dist_path = idx_to_dist(idx_path)
        return NbrInfo(
            name,
            n_nbrs,
            metric,
            exact,
            method,
            dist_path.exists(),
            idx_path,
            dist_path,
        )


@dataclass
class NearestNeighbors:
    """Nearest neighbors information. Contains the indices and optionally the distances
    to the neighbors. The `info` field contains information about the neighbors.

    The `idx` attribute is a numpy array of shape (n_samples, n_neighbors) and contains
    in each row the indices of the neighbors of the corresponding sample.

    The `dist` attribute, if present, is a numpy array of shape (n_samples, n_neighbors)
    and contains in each row the distances to the neighbors of the corresponding sample.
    """

    idx: np.ndarray
    info: NbrInfo | None = None
    dist: np.ndarray | None = None


def replace_n_neighbors_in_path(path: Path, n_neighbors: int) -> Path:
    """Replace the number of neighbors in the path.
    e.g. iris.15.euclidean.exact.faiss.idx.npy => iris.65.euclidean.exact.faiss.idx.npy
    """
    parts = path.name.split(".")
    parts[1] = str(n_neighbors)
    new_name = ".".join(parts)
    return path.parent / new_name
