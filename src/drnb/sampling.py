from typing import Literal

import numpy as np


def ncvis_negative_plan(n_negative: int = 5, n_epochs: int = 200) -> np.ndarray:
    """Create a plan for the number of negative samples to use in each epoch.
    Sampling in the ncvis style of slowly increasing the number of negatives over
    epochs (but on average sticking to `n_negative`)
    """
    negative_plan = np.linspace(0, 1, n_epochs)
    negative_plan = negative_plan**3

    negative_plan /= negative_plan.sum()
    negative_plan *= n_epochs * n_negative
    negative_plan = negative_plan.round().astype(int)
    negative_plan[negative_plan < 1] = 1
    return negative_plan


def create_sample_plan(
    n_samples: int,
    n_epochs: int,
    strategy: Literal["unif", "inc", "dec"] | None = "unif",
    n_obs=int | None,
) -> np.ndarray:
    """Create a plan for the number of samples to use in each epoch. The plan can
    be uniform ("unif"), increasing ("inc"), or decreasing ("dec"). If `n_obs` is not
    None, the plan is clipped to have at most `n_obs` samples in each epoch."""
    if strategy is None:
        strategy = "unif"

    if strategy == "inc":
        samples = ncvis_negative_plan(n_samples, n_epochs)
    elif strategy == "dec":
        samples = np.flip(ncvis_negative_plan(n_samples, n_epochs))
    elif strategy == "unif":
        samples = np.array([n_samples] * n_epochs, dtype=int)
    else:
        raise ValueError(f"Unknown sample strategy {strategy}")
    if n_obs is not None:
        samples[samples > n_obs] = n_obs
    return samples
