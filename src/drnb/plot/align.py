import numpy as np


def kabsch(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Compute the optimal rotation matrix using the Kabsch algorithm.

    Parameters
    ----------
    fixed : np.ndarray
        The reference/target coordinates that stay fixed
    moving : np.ndarray
        The source coordinates that will be transformed
    """
    centroid_fixed = np.mean(fixed, axis=0)
    centroid_moving = np.mean(moving, axis=0)

    # Center the points
    fixed_centered = fixed - centroid_fixed
    moving_centered = moving - centroid_moving

    # Compute the covariance matrix
    H = np.dot(moving_centered.T, fixed_centered)
    U, _, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    Vt[-1, :] *= d
    U[:, -1] *= d
    rotation = np.dot(Vt.T, U.T)

    return rotation


def rmsd(fixed: np.ndarray, transformed: np.ndarray) -> float:
    """Compute the Root Mean Square Deviation between two point sets."""
    return np.sqrt(np.mean(np.sum((fixed - transformed) ** 2, axis=1)))


def transform_points(
    fixed: np.ndarray, moving: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Transform the moving points to align with the fixed points using the given
    rotation matrix."""
    transformed = np.dot(moving - np.mean(moving, axis=0), rotation) + np.mean(
        fixed, axis=0
    )
    return transformed


def kabsch_align(
    fixed: np.ndarray,
    moving: np.ndarray,
    return_rmsd: bool = False,
) -> tuple[np.ndarray, float] | np.ndarray:
    """Align the moving points to the fixed points using the Kabsch algorithm.

    Parameters
    ----------
    fixed : np.ndarray
        The reference/target coordinates that stay fixed
    moving : np.ndarray
        The source coordinates that will be transformed
    return_rmsd : bool
        If True, return the RMSD value along with transformed coordinates

    Returns
    -------
    transformed : np.ndarray
        The aligned coordinates
    rmsd_value : float, optional
        The RMSD between fixed and transformed coordinates if return_rmsd=True
    """

    rotation = kabsch(fixed, moving)
    transformed = transform_points(fixed, moving, rotation)

    if return_rmsd:
        rmsd_value = rmsd(fixed, transformed)
        return transformed, rmsd_value

    return transformed


def kabsch_best_align(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> np.ndarray:
    """Find the best alignment between fixed and moving points, considering reflection.

    Tries both original and reflected versions of moving points and returns the
    alignment with lower RMSD.

    Parameters
    ----------
    fixed : np.ndarray
        The reference/target coordinates that stay fixed
    moving : np.ndarray
        The source coordinates that will be transformed

    Returns
    -------
    np.ndarray
        The best aligned coordinates
    """
    # Try original alignment
    transformed, rmsd_orig = kabsch_align(fixed, moving, return_rmsd=True)

    # Try reflected alignment
    moving_reflected = moving * np.array([-1, 1])
    transformed_reflected, rmsd_reflected = kabsch_align(
        fixed, moving_reflected, return_rmsd=True
    )

    return transformed_reflected if rmsd_reflected < rmsd_orig else transformed
