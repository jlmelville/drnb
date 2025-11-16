from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np


@dataclass
class NbrInfo:
    name: str
    n_nbrs: int
    metric: str
    exact: bool
    method: str
    has_distances: bool
    idx_path: Path | None
    dist_path: Path | None

    @property
    def neighbor_type(self) -> str:
        return "exact" if self.exact else "approximate"

    @property
    def idx_suffix(self) -> str:
        return nn_suffix(self.idx_path) if self.idx_path else self._suffix("idx")

    @property
    def dist_suffix(self) -> str:
        return nn_suffix(self.dist_path) if self.dist_path else self._suffix("dist")

    def _suffix(self, nn_component: str) -> str:
        return "." + ".".join(
            [
                str(self.n_nbrs),
                self.metric,
                self.neighbor_type,
                self.method,
                nn_component,
            ]
        )


@dataclass
class NearestNeighbors:
    idx: np.ndarray
    info: NbrInfo | None = None
    dist: np.ndarray | None = None


def idx_to_dist(idx_path: Path) -> Path:
    dist_stem = idx_path.stem[:-3] + "dist"
    return idx_path.parent / (dist_stem + idx_path.suffix)


def nn_suffix(path: Path | None) -> str:
    if path is None:
        return ""
    return "".join(path.suffixes[:-1])


def replace_n_neighbors_in_path(path: Path, n_neighbors: int) -> Path:
    parts = path.name.split(".")
    parts[1] = str(n_neighbors)
    return path.parent / ".".join(parts)


def find_candidate_neighbors_info(
    base_dir: Path,
    name: str,
    *,
    n_neighbors: int | None = None,
    metric: str,
    method: str | None = None,
    exact: bool | None = None,
    require_distance: bool = True,
) -> NbrInfo | None:
    """Find the candidate neighbors info for the given name, metric, method, and exact.
    `exact`, `method`, and `require_distance` are optional filters. If they are not
    None, then only those results that match all of them are returned.

    If `exact` is None, then exact neighbors are preferred if available, otherwise
    approximate neighbors are returned.

    If `n_neighbors` is specified then the result with the smallest number of neighbors
    greater than or equal to `n_neighbors` is returned. If `n_neighbors` is None, the
    result with the maximum number of neighbors is returned. The latter case is for
    when we don't know how many neighbors to use at the point of providing the file
    (e.g. with a plugin embedder with internal logic that will ask for a larger number
    of neighbors based on the base `n_neighbors`.

    If there are no results that match the filters, None is returned. Otherwise, the
    best result is returned.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    pattern = f"{name}.*.idx.*"
    files = list(base_dir.glob(pattern))
    nn_infos: list[NbrInfo] = []
    for file in files:
        info = _nbr_info_from_path(file)
        if info is not None and info.metric == metric:
            nn_infos.append(info)
    if not nn_infos:
        return None
    if exact is None:
        # if no exact preference is specified, prefer exact neighbors
        exact_nn_infos = [info for info in nn_infos if info.exact]
        if exact_nn_infos:
            nn_infos = exact_nn_infos
        else:
            # but it's not a failure if we don't have any exact neighbors so take
            # the approximate neighbors instead
            nn_infos = [info for info in nn_infos if not info.exact]
    else:
        nn_infos = [info for info in nn_infos if info.exact is exact]
    if method is not None:
        nn_infos = [info for info in nn_infos if info.method == method]
    if require_distance:
        nn_infos = [info for info in nn_infos if info.has_distances]
    if n_neighbors is None:
        # get the result with the maximum number of neighbors
        max_n_neighbors = max(info.n_nbrs for info in nn_infos)
        nn_infos = [info for info in nn_infos if info.n_nbrs == max_n_neighbors]
    else:
        nn_infos = [info for info in nn_infos if info.n_nbrs >= n_neighbors]
    if not nn_infos:
        return None
    nn_infos = cast(list[NbrInfo], nn_infos)
    candidate_infos = sorted(nn_infos, key=lambda x: x.n_nbrs)
    preferred_exts = [".npy", ".pkl"]
    for info in candidate_infos:
        if info.idx_path is not None and info.idx_path.suffix in preferred_exts:
            return info
    return candidate_infos[0]


def read_neighbors(
    base_dir: Path,
    name: str,
    *,
    n_neighbors: int,
    metric: str,
    method: str | None = None,
    exact: bool | None = None,
    require_distance: bool = True,
) -> NearestNeighbors | None:
    info = find_candidate_neighbors_info(
        base_dir,
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        exact=exact,
        require_distance=require_distance,
    )
    if info is None:
        return None
    idx = _load_array(info.idx_path)[:, :n_neighbors]
    dist = None
    if require_distance:
        dist = _load_array(info.dist_path)[:, :n_neighbors]
    return NearestNeighbors(idx=idx, dist=dist, info=info)


def write_neighbors(
    base_dir: Path,
    neighbor_data: NearestNeighbors,
    *,
    file_types: Sequence[str] = ("npy",),
) -> tuple[list[Path], list[Path]]:
    if neighbor_data.info is None:
        raise ValueError("Cannot write neighbors without metadata")
    idx_paths = _write_array(
        base_dir,
        neighbor_data.info.name,
        neighbor_data.info.idx_suffix,
        neighbor_data.idx,
        file_types,
    )
    dist_paths: list[Path] = []
    if neighbor_data.dist is not None:
        dist_paths = _write_array(
            base_dir,
            neighbor_data.info.name,
            neighbor_data.info.dist_suffix,
            neighbor_data.dist,
            file_types,
        )
    return idx_paths, dist_paths


def _nbr_info_from_path(path: Path) -> NbrInfo | None:
    items = path.stem.split(".")
    if len(items) != 6:
        return None
    name = items[0]
    n_nbrs = int(items[1])
    metric = items[2]
    if items[3] == "exact":
        exact = True
    elif items[3] == "approximate":
        exact = False
    else:
        return None
    method = items[4]
    dist_path = idx_to_dist(path)
    return NbrInfo(
        name=name,
        n_nbrs=n_nbrs,
        metric=metric,
        exact=exact,
        method=method,
        has_distances=dist_path.exists(),
        idx_path=path,
        dist_path=dist_path if dist_path.exists() else None,
    )


def _load_array(path: Path | None) -> np.ndarray:
    if path is None:
        raise FileNotFoundError("Neighbor file missing")
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return data[data.files[0]]
    if path.suffix == ".pkl":
        with path.open("rb") as fh:
            return pickle.load(fh)
    raise ValueError(f"Unsupported neighbor file type: {path.suffix}")


def _write_array(
    base_dir: Path,
    name: str,
    suffix: str,
    array: np.ndarray,
    file_types: Sequence[str],
) -> list[Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    array = np.asarray(array)
    for file_type in file_types:
        filename = f"{name}{suffix}.{file_type}"
        path = base_dir / filename
        if file_type == "npy":
            np.save(path, array)
        elif file_type == "npz":
            np.savez_compressed(path, array)
        elif file_type == "pkl":
            with path.open("wb") as fh:
                pickle.dump(array, fh)
        else:
            raise ValueError(f"Unsupported file type {file_type}")
        paths.append(path)
    return paths
