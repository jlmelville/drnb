from __future__ import annotations

from pathlib import Path
from typing import List, cast

import numpy as np

from drnb.io import data_relative_path, get_path, read_data, write_data
from drnb.log import log
from drnb.neighbors.nbrinfo import NbrInfo, NearestNeighbors


# pylint: disable=too-many-return-statements
def find_candidate_neighbors_info(
    name: str | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    n_neighbors: int = 1,
    metric: str = "euclidean",
    method: str | None = None,
    exact: bool | None = None,
    return_distance: bool = True,
    verbose: bool = False,
) -> NbrInfo | None:
    """Find the most suitable pre-calculated nearest neighbors info for a dataset."""
    if name is None:
        if verbose:
            log.warning("No name provided to find candidate neighbors info")
        return None

    try:
        nn_dir_path = get_path(drnb_home=drnb_home, sub_dir=sub_dir)
    except FileNotFoundError as e:
        if verbose:
            log.warning("No neighbors directory found at %s", e.filename)
        return None

    nn_file_paths = list(Path.glob(nn_dir_path, name + ".*.idx.*"))
    if not nn_file_paths:
        if verbose:
            log.warning("No neighbors files found for %s", name)
        return None

    nn_infos = [
        NbrInfo.from_path(nn_file_path, ignore_bad_path=True)
        for nn_file_path in nn_file_paths
        if nn_file_path.stem.startswith(name)
    ]
    nn_infos = [nn_info for nn_info in nn_infos if nn_info is not None]
    if not nn_infos:
        if verbose:
            log.warning("No neighbors info found for %s", name)
        return None

    nn_infos = [nn_info for nn_info in nn_infos if nn_info.metric == metric]
    if not nn_infos:
        if verbose:
            log.warning("No neighbors info found for %s with metric %s", name, metric)
        return None

    if exact is not None:
        if exact:
            nn_infos = [nn_info for nn_info in nn_infos if nn_info.exact]
        else:
            nn_infos = [nn_info for nn_info in nn_infos if not nn_info.exact]
        if not nn_infos:
            if verbose:
                log.warning("No neighbors info found for %s with exact=%s", name, exact)
            return None

    if method is not None:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.method == method]
        if not nn_infos:
            if verbose:
                log.warning(
                    "No neighbors info found for %s with method %s", name, method
                )
            return None

    if return_distance:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.has_distances]
        if not nn_infos:
            if verbose:
                log.warning(
                    "No neighbors info found for %s with return_distance=%s",
                    name,
                    return_distance,
                )
            return None

    nn_infos = [nn_info for nn_info in nn_infos if nn_info.n_nbrs >= n_neighbors]
    if not nn_infos:
        if verbose:
            log.warning(
                "No neighbors info found for %s with n_neighbors=%d", name, n_neighbors
            )
        return None

    nn_infos = cast(List[NbrInfo], nn_infos)
    candidate_infos = sorted(nn_infos, key=lambda x: x.n_nbrs)
    candidate_info = candidate_infos[0]

    preferred_exts = [".npy", ".pkl"]
    for cinfo in candidate_infos:
        if cinfo.idx_path is not None and cinfo.idx_path.suffix in preferred_exts:
            if verbose:
                log.info("Found pre-calculated neighbors file: %s", cinfo.idx_path)
            return cinfo
    if verbose:
        log.info("No suitable pre-calculated neighbors available")
    return candidate_info


def read_neighbors(
    name: str,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    method: str | None = None,
    exact: bool | None = False,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    return_distance: bool = True,
    verbose: bool = False,
) -> NearestNeighbors | None:
    """Read pre-calculated nearest neighbors from disk."""
    candidate_info = find_candidate_neighbors_info(
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        exact=exact,
        return_distance=return_distance,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
    )
    if candidate_info is None:
        if verbose:
            log.info("No suitable pre-calculated neighbors available")
        return None

    if verbose:
        log.info(
            "Found pre-calculated neighbors file: %s",
            data_relative_path(candidate_info.idx_path),
        )
    idx = read_data(
        dataset=candidate_info.name,
        suffix=candidate_info.idx_suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        as_numpy=True,
        verbose=False,
    )
    idx = idx[:, :n_neighbors]
    if return_distance:
        dist = read_data(
            dataset=candidate_info.name,
            suffix=candidate_info.dist_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            as_numpy=True,
            verbose=False,
        )
        dist = dist[:, :n_neighbors]
        return NearestNeighbors(idx=idx, dist=dist, info=candidate_info)
    return NearestNeighbors(idx=idx, dist=None, info=candidate_info)


def write_neighbors(
    neighbor_data: NearestNeighbors,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    create_sub_dir: bool = False,
    file_type: str | List[str] = "pkl",
    verbose: bool = False,
) -> tuple[List[Path], List[Path]]:
    """Write nearest neighbors to disk."""
    if neighbor_data.info is None:
        raise ValueError("Cannot write neighbors without NbrInfo metadata")

    if isinstance(file_type, str):
        file_type = [file_type]

    idx_paths = write_data(
        x=neighbor_data.idx,
        name=neighbor_data.info.name,
        suffix=neighbor_data.info.idx_suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
        file_type=file_type,
    )
    dist_paths: list[Path] = []
    if neighbor_data.dist is not None:
        dist_paths = write_data(
            x=neighbor_data.dist,
            name=neighbor_data.info.name,
            suffix=neighbor_data.info.dist_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
            file_type=file_type,
        )
    return idx_paths, dist_paths
