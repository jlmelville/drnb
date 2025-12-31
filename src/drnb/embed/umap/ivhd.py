import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom import CustomGradientUMAP, epoch_func
from drnb.types import EmbedResult


class IvhdGradientArgs(NamedTuple):
    """Gradient arguments for IVHD.

    Attributes:
        gamma: float: The repulsion strength.
        near_dist: float: The near distance to use for near neighbor pairs.
        far_dist: float: The far distance to use for far pairs.
    """

    gamma: float = 0.01
    near_dist: float = 0.0
    far_dist: float = 1.0


def ivhd_grad_coeff_attr(_, __) -> float:
    """Gradient coefficient for the attractive cost function in IVHD."""
    return -2.0


def ivhd_grad_coeff_rep(d2: float, grad_args: IvhdGradientArgs) -> float:
    """Gradient coefficient for the repulsive cost function in IVHD."""
    gamma = grad_args.gamma
    r = grad_args.far_dist
    d = math.sqrt(d2)

    grad_coeff = 2.0 * gamma * r
    grad_coeff /= 0.001 + d
    return grad_coeff


class IVHD(CustomGradientUMAP):
    """IVHD implementation."""

    def get_gradient_args(self):
        return IvhdGradientArgs(
            gamma=self.repulsion_strength,
            near_dist=self.near_dist,
            far_dist=self.far_dist,
        )

    def __init__(self, near_dist=0.0, far_dist=1.0, **kwargs):
        self.near_dist = near_dist
        self.far_dist = far_dist

        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True

        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=ivhd_grad_coeff_attr,
            custom_rep_func=ivhd_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Ivhd(drnb.embed.umap.Umap):
    """Embedder for IVHD."""

    use_precomputed_knn: bool = True
    drnb_init: str | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, IVHD, "ivhd")
