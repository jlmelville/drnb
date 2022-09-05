from dataclasses import dataclass
from pathlib import Path

import numpy as np

import drnb.io as nbio
from drnb.log import log


def get_triplets(X, seed=None, n_triplets_per_point=5):
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate n_triplets_per_point pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], n_triplets_per_point, 2))
    return triplets


def calc_distances(X, pairs):
    distances = np.empty(pairs.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u, v) in pairs]
    return distances


def validate_triplets(triplets, n_obs):
    if len(triplets.shape) != 3 or triplets.shape[2] != 2 or triplets.shape[0] != n_obs:
        raise ValueError(
            f"triplets should have shape ({n_obs}, n_triplets_per_point, 2)"
        )


def find_triplet_files(
    name, n_triplets_per_point=5, metric="l2", data_path=None, sub_dir="triplets"
):
    triplet_dir_path = nbio.get_data_path(data_path=data_path, sub_dir=sub_dir)
    if not triplet_dir_path.exists():
        return []
    triplet_file_paths = list(Path.glob(triplet_dir_path, name + "*.idx.*"))
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

    # look for suitable distance files
    dist_triplet_infos = [
        info for info in triplet_infos if info.dist_path(metric).exists()
    ]
    # if there weren't any then just the triplets will do
    if not dist_triplet_infos:
        return triplet_infos
    return dist_triplet_infos


def write_triplets(
    idx,
    name,
    n_triplets_per_point,
    seed,
    data_path=None,
    sub_dir="triplets",
    create_sub_dir=True,
    verbose=False,
    dist=None,
    metric="l2",
):
    if idx.shape[1] != n_triplets_per_point:
        raise ValueError(
            "triplet data should have "
            + f"{n_triplets_per_point} triplets per point, but was {idx.shape[1]}"
        )

    # e.g. mnist.5.42.idx.npy
    nbio.write_npy(
        idx,
        name,
        suffix=f".{n_triplets_per_point}.{seed}.idx",
        data_path=data_path,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
    )
    # e.g. mnist.5.42.l2.npy
    if dist is not None:
        nbio.write_npy(
            dist,
            name,
            suffix=f".{n_triplets_per_point}.{seed}.{metric}",
            data_path=data_path,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
        )


def read_triplets_from_info(triplet_info, metric="l2", verbose=False):
    idx = nbio.read_data(
        dataset=triplet_info.name,
        suffix=triplet_info.idx_suffix,
        sub_dir=None,
        data_path=triplet_info.idx_path.parent,
        as_numpy=np.int64,
        verbose=verbose,
    )
    dist = None
    if triplet_info.dist_path(metric).exists():
        dist = nbio.read_data(
            dataset=triplet_info.name,
            suffix=triplet_info.dist_suffix(metric),
            sub_dir=None,
            data_path=triplet_info.idx_path.parent,
            as_numpy=True,
            verbose=verbose,
        )
    return idx, dist


def find_precomputed_triplets(ctx, n_triplets_per_point, metric):
    log.info("Looking for precomputed triplets")
    triplet_infos = find_triplet_files(
        name=ctx.name,
        n_triplets_per_point=n_triplets_per_point,
        data_path=ctx.data_path,
        sub_dir=ctx.triplet_sub_dir,
    )
    if not triplet_infos:
        return None, None, True

    triplet_info = triplet_infos[0]
    log.info("Using triplets from %s", nbio.data_relative_path(triplet_info.idx_path))
    idx, dist = read_triplets_from_info(triplet_info=triplet_info)
    if dist is not None:
        log.info("Also found corresponding %s distances", metric)

    # if we wanted pre-computed triplets but couldn't get them and now have
    # the opportunity to cache them, then return the triplet data
    return_triplets = idx is None or dist is None
    return idx, dist, return_triplets


def cache_triplets(idx, dist, ctx, n_triplets_per_point, metric, random_state):
    log.info("Caching triplets")
    write_triplets(
        idx,
        name=ctx.name,
        n_triplets_per_point=n_triplets_per_point,
        seed=random_state,
        data_path=ctx.data_path,
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
