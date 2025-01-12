from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Self

import numpy as np
from numba import jit, prange

import drnb.io as nbio
from drnb.distances import distance_function
from drnb.log import log
from drnb.types import DistanceFunc
from drnb.util import FromDict, Jsonizable, islisty


def calculate_triplets(
    data: np.ndarray,
    seed: int = None,
    n_triplets_per_point: int = 5,
    return_distance: bool = True,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Generate triplets for a dataset X. Each row of X is a point in the dataset.
    Returns a 3D array of indices into X, with shape (X.shape[0], n_triplets_per_point,
    2). If return_distance is True, also returns a 3D array of distances with the same
    shape as the idx array."""
    idx = get_triplets(X=data, seed=seed, n_triplets_per_point=n_triplets_per_point)
    if return_distance:
        dist_fun = distance_function(metric)
        dist = calc_distances(data, idx, dist_fun)
        return idx, dist
    return idx


def get_triplets(
    X: np.ndarray, seed: int = None, n_triplets_per_point: int = 5
) -> np.ndarray:
    """Generate triplets for a dataset X. Each row of X is a point in the dataset.
    Returns a 3D array of indices into X, with shape (X.shape[0], n_triplets_per_point,
    2).
    """
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate n_triplets_per_point pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], n_triplets_per_point, 2))
    return triplets


@jit(nopython=True, parallel=True)
def calc_distances(
    X: np.ndarray, idx: np.ndarray, dist_fun: DistanceFunc
) -> np.ndarray:
    """Calculate distances between triplets of points. idx is a 3D array of indices
    into X, with shape (n_items, n_triplets_per_item, 2). The output is a 3D array
    of distances with the same shape as the idx array."""
    n_items, n_triplets_per_item, n_others = idx.shape
    distances = np.empty(idx.shape, dtype=np.float32)
    # pylint: disable=not-an-iterable
    for i in prange(n_items):
        item_pairs = idx[i]
        for j in range(n_triplets_per_item):
            pair = item_pairs[j]
            for k in range(n_others):
                distances[i, j, k] = dist_fun(X[i], X[pair[k]])
    return distances


def validate_triplets(triplets: np.ndarray, n_obs: int):
    """Check that the triplets array has the correct shape."""
    if len(triplets.shape) != 3 or triplets.shape[2] != 2 or triplets.shape[0] != n_obs:
        raise ValueError(
            f"triplets should have shape ({n_obs}, n_triplets_per_point, 2)"
        )


# e.g. mnist.5.42.idx.npy
@dataclass
class TripletInfo:
    """Information about a set of triplets."""

    name: str  # name of the dataset
    n_triplets_per_point: int  # number of triplets per point
    seed: int  # random seed that generated the triplets
    idx_path: Path  # path to triplet file

    @property
    def idx_suffix(self) -> str:
        """Return the suffix for the idx file."""
        return self._suffix("idx")

    def dist_suffix(self, metric) -> str:
        """Return the suffix for the distance file."""
        return self._suffix(metric)

    def _suffix(self, triplet_kind: str) -> str:
        """Return the suffix for the idx or distance file."""
        return "." + ".".join(
            [
                str(self.n_triplets_per_point),
                str(self.seed),
                triplet_kind,
            ]
        )

    def dist_path(self, metric: str = "l2") -> Path:
        """Return the path to the distance file."""
        components = self.idx_path.name.split(".")
        components[3] = metric
        return self.idx_path.parent / ".".join(components)

    @classmethod
    def from_path(
        cls, triplet_path: Path, ignore_bad_path: bool = False
    ) -> Self | None:
        """Create a TripletInfo object from a triplet file path."""
        items = triplet_path.stem.split(".")
        if len(items) != 4:
            msg = f"Unknown triplet file format: {triplet_path}"
            if ignore_bad_path:
                log.warning(msg)
                return None
            raise ValueError(msg)
        name = items[0]
        n_triplets_per_point = int(items[1])
        seed = int(items[2])

        return TripletInfo(name, n_triplets_per_point, seed, triplet_path)


# pylint:disable=too-many-return-statements
def find_triplet_files(
    name: str,
    n_triplets_per_point: int = 5,
    metric: str = "l2",
    drnb_home: Path | str | None = None,
    sub_dir: str = "triplets",
    seed: int = None,
) -> List[TripletInfo]:
    """Find triplet files for a dataset. Returns a list of TripletInfo objects.
    If seed is not None, only returns files with that seed. If metric is not None,
    only returns files with that metric. If no files are found, returns an empty list."""
    try:
        triplet_dir_path = nbio.get_path(drnb_home=drnb_home, sub_dir=sub_dir)
    except FileNotFoundError:
        return []

    if not triplet_dir_path.exists():
        return []
    triplet_file_paths = list(Path.glob(triplet_dir_path, name + ".*.idx.*"))
    triplet_infos = [
        TripletInfo.from_path(triplet_file_path, ignore_bad_path=True)
        for triplet_file_path in triplet_file_paths
        if triplet_file_path.stem.startswith(name)
    ]
    triplet_infos = [info for info in triplet_infos if info is not None]
    if not triplet_infos:
        return triplet_infos

    triplet_infos = [
        info
        for info in triplet_infos
        if info.n_triplets_per_point == n_triplets_per_point
    ]
    if not triplet_infos:
        return triplet_infos

    if seed is not None:
        triplet_infos = [info for info in triplet_infos if info.seed == seed]
        if not triplet_infos:
            return triplet_infos

    # favor npy or pkl files over csv
    preferred_exts = [".npy", ".pkl"]
    ext_triplet_infos = [
        info for info in triplet_infos if info.idx_path.suffix in preferred_exts
    ]
    if ext_triplet_infos:
        triplet_infos = ext_triplet_infos

    # look for suitable distance files
    dist_triplet_infos = [
        info for info in triplet_infos if info.dist_path(metric).exists()
    ]
    # if there weren't any then just the triplets will do
    if not dist_triplet_infos:
        return triplet_infos
    return dist_triplet_infos


# flattened = True means idx and dist have been turned into 1D arrays so they can
# be exported as e.g. CSV
def write_triplets(
    idx: np.ndarray,
    name: str,
    n_triplets_per_point: int,
    seed: int,
    drnb_home: Path | str | None = None,
    sub_dir: str = "triplets",
    create_sub_dir: bool = True,
    file_type: str | List[str] = "npy",
    verbose: bool = False,
    dist: np.ndarray | None = None,
    metric: str = "euclidean",
    flattened: bool = False,
    suffix: str | List[str] | None = None,
) -> tuple[List[Path], List[Path]]:
    """Write triplet data to files. Returns a tuple of paths to the idx and dist files.
    If dist is None, the second element of the tuple will be an empty list."""
    if not flattened and idx.shape[1] != n_triplets_per_point:
        raise ValueError(
            "triplet data should have "
            + f"{n_triplets_per_point} triplets per point, but was {idx.shape[1]}"
        )

    idx_suffix = f".{n_triplets_per_point}.{seed}.idx"
    if suffix is not None:
        idx_suffix = nbio.ensure_suffix(suffix) + idx_suffix
    # e.g. mnist.5.42.idx.npy
    idx_paths = nbio.write_data(
        idx,
        name,
        suffix=idx_suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
        file_type=file_type,
    )
    # e.g. mnist.5.42.l2.npy

    dist_suffix = f".{n_triplets_per_point}.{seed}.{metric}"
    if suffix is not None:
        dist_suffix = nbio.ensure_suffix(suffix) + dist_suffix
    dist_paths = []
    if dist is not None:
        dist_paths = nbio.write_data(
            dist,
            name,
            suffix=dist_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
            file_type=file_type,
        )
    return idx_paths, dist_paths


def read_triplets_from_info(
    triplet_info: TripletInfo, metric: str = "l2", verbose: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read triplets from a TripletInfo object. Returns a tuple of (idx, dist).
    If no distance file is found, dist is None."""
    idx = nbio.read_data(
        dataset=triplet_info.name,
        suffix=triplet_info.idx_suffix,
        sub_dir=None,
        drnb_home=triplet_info.idx_path.parent,
        as_numpy=np.int64,
        verbose=verbose,
    )
    dist = None
    if triplet_info.dist_path(metric).exists():
        dist = nbio.read_data(
            dataset=triplet_info.name,
            suffix=triplet_info.dist_suffix(metric),
            sub_dir=None,
            drnb_home=triplet_info.idx_path.parent,
            as_numpy=True,
            verbose=verbose,
        )
    return idx, dist


def find_precomputed_triplets(
    dataset_name: str,
    triplet_sub_dir: str,
    n_triplets_per_point: int,
    metric: str = "euclidean",
    drnb_home: Path | str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find precomputed triplets for a dataset. Returns a tuple of (idx, dist).
    If no triplets are found, returns (None, None)."""
    log.info("Looking for precomputed triplets")
    triplet_infos = find_triplet_files(
        name=dataset_name,
        n_triplets_per_point=n_triplets_per_point,
        drnb_home=drnb_home,
        sub_dir=triplet_sub_dir,
        metric=metric,
    )
    if not triplet_infos:
        return None, None

    triplet_info = triplet_infos[0]
    log.info("Using triplets from %s", nbio.data_relative_path(triplet_info.idx_path))
    idx, dist = read_triplets_from_info(triplet_info=triplet_info, metric=metric)
    if dist is not None:
        log.info("Also found corresponding %s distances", metric)

    return idx, dist


@dataclass
class TripletsRequest(FromDict, Jsonizable):
    """Request for creating triplets."""

    n_triplets_per_point: int = 5
    seed: int = 42
    file_types: list = field(default_factory=lambda: ["pkl"])
    metric: list = field(default_factory=lambda: ["euclidean"])

    def create_triplets(
        self,
        data: np.ndarray,
        dataset_name: str,
        triplet_dir: str,
        suffix: str | List[str] | None = None,
    ) -> List[Path]:
        """Create triplets for a dataset. Returns a list of paths to the idx and dist
        files."""
        if not islisty(self.metric):
            metrics = [self.metric]
        else:
            metrics = self.metric

        triplet_output_paths = []
        for metric in metrics:
            idx, dist = calculate_triplets(
                data,
                seed=self.seed,
                n_triplets_per_point=self.n_triplets_per_point,
                return_distance=True,
                metric=metric,
            )

            file_types = self.file_types
            if "csv" in file_types:
                # treat CSV specially because we need to flatten distances
                file_types = [ft for ft in self.file_types if ft != "csv"]
                csv_idx_paths, csv_dist_paths = write_triplets(
                    idx.flatten(),
                    name=dataset_name,
                    n_triplets_per_point=self.n_triplets_per_point,
                    seed=self.seed,
                    sub_dir=triplet_dir,
                    create_sub_dir=True,
                    file_type="csv",
                    verbose=True,
                    dist=dist.flatten(),
                    flattened=True,
                    metric=metric,
                    suffix=suffix,
                )
                triplet_output_paths += csv_idx_paths + csv_dist_paths

            triplet_idx_paths, triplet_dist_paths = write_triplets(
                idx,
                dataset_name,
                self.n_triplets_per_point,
                self.seed,
                sub_dir=triplet_dir,
                create_sub_dir=True,
                file_type=file_types,
                verbose=True,
                dist=dist,
                metric=metric,
                suffix=suffix,
            )
            triplet_output_paths += triplet_idx_paths + triplet_dist_paths
        return triplet_output_paths


# triplets = (
#   n_triplets_per_point=5,
#   seed=42,
def create_triplets_request(triplets_kwds: dict | None) -> TripletsRequest | None:
    """Create a TripletsRequest based on the provided keyword arguments."""
    if triplets_kwds is None:
        return None
    return TripletsRequest.new(**triplets_kwds)
