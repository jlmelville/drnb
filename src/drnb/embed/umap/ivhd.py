import math
from dataclasses import dataclass
from typing import NamedTuple

import drnb.embed.umap
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.log import log


# pylint: disable=unused-argument
def ivhd_grad_coeff_attr(d2, grad_args):
    return -2.0


def ivhd_grad_coeff_rep(d2, grad_args):
    gamma = grad_args.gamma
    r = grad_args.far_dist
    d = math.sqrt(d2)

    grad_coeff = 2.0 * gamma * r
    grad_coeff /= 0.001 + d
    return grad_coeff


class IVHD(CustomGradientUMAP2):
    def get_gradient_args(self):
        class IvhdGradientArgs(NamedTuple):
            gamma: float = 0.01
            near_dist: float = 0.0
            far_dist: float = 1.0

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
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return embed_ivhd(x, params)


def embed_ivhd(
    x,
    params,
):
    log.info("Running ivhd")
    embedder = IVHD(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
