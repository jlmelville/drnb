import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from drnb.log import log


def idx_to_dist(idx_path):
    dist_stem = idx_path.stem[:-3] + "dist"
    dist_path = idx_path.parent / (dist_stem + idx_path.suffix)
    return dist_path


# everything between name and the extension (exclusive)
# e.g. iris.15.euclidean.exact.faiss.idx.npy => .15.euclidean.exact.faiss.idx
def nn_suffix(path):
    return "".join(path.suffixes[:-1])


@dataclass
class NbrInfo:
    name: str  # name of the dataset
    n_nbrs: int  # number of neighbors
    metric: str  # e.g. "euclidean"
    exact: bool  # exact or approximate neighbors
    method: str  # method to generate e.g. "faiss", "annoy", "hnsw"
    has_distances: bool  # True if the distances are present
    idx_path: Optional[pathlib.Path]  # the path to the idx file (optional)
    dist_path: Optional[pathlib.Path]  # the path to the dist file (if it exists)

    @property
    def idx_suffix(self):
        if self.idx_path is not None:
            return nn_suffix(self.idx_path)
        return self.suffix("idx")

    @property
    def dist_suffix(self):
        if self.idx_path is not None:
            return nn_suffix(self.dist_path)
        return self.suffix("dist")

    def suffix(self, nn_component):
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
    def neighbor_type(self):
        if self.exact:
            return "exact"
        return "approximate"

    @classmethod
    def from_path(cls, idx_path, ignore_bad_path=False):
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
    idx: np.ndarray
    info: Optional[NbrInfo] = None
    dist: Optional[np.ndarray] = None
