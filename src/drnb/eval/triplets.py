from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import jit, prange

import drnb.io as nbio
from drnb.distance import distance_function
from drnb.log import log
from drnb.util import FromDict


def calculate_triplets(
    data, seed=None, n_triplets_per_point=5, return_distance=True, metric="euclidean"
):
    idx = get_triplets(X=data, seed=seed, n_triplets_per_point=n_triplets_per_point)
    if return_distance:
        dist_fun = distance_function(metric)
        dist = calc_distances(data, idx, dist_fun)
        return idx, dist
    return idx


def get_triplets(X, seed=None, n_triplets_per_point=5):
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate n_triplets_per_point pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], n_triplets_per_point, 2))
    return triplets


@jit(nopython=True, parallel=True)
def calc_distances(X, idx, dist_fun):
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


def validate_triplets(triplets, n_obs):
    if len(triplets.shape) != 3 or triplets.shape[2] != 2 or triplets.shape[0] != n_obs:
        raise ValueError(
            f"triplets should have shape ({n_obs}, n_triplets_per_point, 2)"
        )


def find_triplet_files(
    name, n_triplets_per_point=5, metric="l2", drnb_home=None, sub_dir="triplets"
):
    triplet_dir_path = nbio.get_path(drnb_home=drnb_home, sub_dir=sub_dir)
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
    idx,
    name,
    n_triplets_per_point,
    seed,
    drnb_home=None,
    sub_dir="triplets",
    create_sub_dir=True,
    file_type="npy",
    verbose=False,
    dist=None,
    metric="l2",
    flattened=False,
):
    if not flattened and idx.shape[1] != n_triplets_per_point:
        raise ValueError(
            "triplet data should have "
            + f"{n_triplets_per_point} triplets per point, but was {idx.shape[1]}"
        )

    # e.g. mnist.5.42.idx.npy
    idx_paths = nbio.write_data(
        idx,
        name,
        suffix=f".{n_triplets_per_point}.{seed}.idx",
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
        file_type=file_type,
    )
    # e.g. mnist.5.42.l2.npy
    dist_paths = []
    if dist is not None:
        dist_paths = nbio.write_data(
            dist,
            name,
            suffix=f".{n_triplets_per_point}.{seed}.{metric}",
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
            file_type=file_type,
        )
    return idx_paths, dist_paths


def read_triplets_from_info(triplet_info, metric="l2", verbose=False):
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


def find_precomputed_triplets(ctx, n_triplets_per_point, metric):
    log.info("Looking for precomputed triplets")
    triplet_infos = find_triplet_files(
        name=ctx.dataset_name,
        n_triplets_per_point=n_triplets_per_point,
        drnb_home=ctx.drnb_home,
        sub_dir=ctx.triplet_sub_dir,
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


def cache_triplets(idx, dist, ctx, n_triplets_per_point, metric, random_state):
    log.info("Caching triplets")
    write_triplets(
        idx,
        name=ctx.dataset_name,
        n_triplets_per_point=n_triplets_per_point,
        seed=random_state,
        drnb_home=ctx.drnb_home,
        sub_dir=ctx.triplet_sub_dir,
        create_sub_dir=True,
        verbose=True,
        dist=dist,
        metric=metric,
    )


# e.g. mnist.5.42.idx.npy
@dataclass
class TripletInfo:
    name: str  # name of the dataset
    n_triplets_per_point: int  # number of triplets per point
    seed: int  # random seed that generated the triplets
    idx_path: Path  # path to triplet file

    @property
    def idx_suffix(self):
        return self.suffix("idx")

    def dist_suffix(self, metric):
        return self.suffix(metric)

    def suffix(self, triplet_kind):
        return "." + ".".join(
            [
                str(self.n_triplets_per_point),
                str(self.seed),
                triplet_kind,
            ]
        )

    def dist_path(self, metric="l2"):
        components = self.idx_path.name.split(".")
        components[3] = metric
        return self.idx_path.parent / ".".join(components)

    @classmethod
    def from_path(cls, triplet_path, ignore_bad_path=False):
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


@dataclass
class TripletsRequest(FromDict):
    n_triplets_per_point: int = 5
    seed: int = 42
    file_types: list = field(default_factory=list)
    metric: list = field(default_factory=lambda: ["euclidean"])
