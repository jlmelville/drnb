from __future__ import annotations

from pathlib import Path

import drnb.neighbors.nbrinfo as nbrinfo
from drnb.io import data_relative_path, get_path, write_data
from drnb.log import log
from drnb.neighbors.nbrinfo import (
    NbrInfo,
    NearestNeighbors,
)


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
    if name is None:
        if verbose:
            log.warning("No name provided to find candidate neighbors info")
        return None
    base_dir = _resolve_base_dir(drnb_home, sub_dir, verbose=verbose)
    if base_dir is None:
        return None
    info = nbrinfo.find_candidate_neighbors_info(
        base_dir,
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        exact=exact,
        require_distance=return_distance,
    )
    if verbose:
        if info is None:
            log.info("No suitable pre-calculated neighbors available")
        else:
            log.info(
                "Found pre-calculated neighbors file: %s",
                data_relative_path(info.idx_path),
            )
    return info


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
    base_dir = _resolve_base_dir(drnb_home, sub_dir, verbose=verbose)
    if base_dir is None:
        return None
    neighbors = nbrinfo.read_neighbors(
        base_dir,
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        exact=exact,
        require_distance=return_distance,
    )
    if neighbors is None and verbose:
        log.info("No suitable pre-calculated neighbors available")
    return neighbors


def write_neighbors(
    neighbor_data: NearestNeighbors,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    create_sub_dir: bool = False,
    file_type: str | list[str] = "pkl",
    verbose: bool = False,
) -> tuple[list[Path], list[Path]]:
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


def _resolve_base_dir(
    drnb_home: Path | str | None, sub_dir: str, verbose: bool = False
) -> Path | None:
    try:
        return get_path(drnb_home=drnb_home, sub_dir=sub_dir)
    except FileNotFoundError as exc:
        if verbose:
            log.warning("No neighbors directory found at %s", exc.filename)
        return None
