from dataclasses import dataclass
from typing import NamedTuple

import drnb.embed.umap
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.log import log


def negtumap_grad_coeff_attr(d2, grad_args):
    gamma = grad_args.gamma

    w = 1 / (1 + d2)
    grad_coeff = -2.0 * gamma * w
    grad_coeff /= w + gamma
    return grad_coeff


def negtumap_grad_coeff_rep(d2, grad_args):
    gamma = grad_args.gamma

    w = 1 / (1 + d2)
    grad_coeff = 2.0 * w * w
    grad_coeff /= w + gamma
    return grad_coeff


class NegTUMAP(CustomGradientUMAP2):
    def get_gradient_args(self):
        class UmapGradientArgs(NamedTuple):
            gamma: float

        return UmapGradientArgs(gamma=self.repulsion_strength)

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True

        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=negtumap_grad_coeff_attr,
            custom_rep_func=negtumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class NegTumap(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return embed_negtumap(x, params)


def embed_negtumap(
    x,
    params,
):
    log.info("Running Neg-t-UMAP")
    embedder = NegTUMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
