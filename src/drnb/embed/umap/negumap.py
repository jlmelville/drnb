from dataclasses import dataclass
from typing import NamedTuple

import drnb.embed.umap
from drnb.embed import run_embed
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func


def negumap_grad_coeff_attr(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    gamma = grad_args.gamma

    d2b = pow(d2, b)
    w = 1 / (1 + a * d2b)
    grad_coeff = -2.0 * a * b * w * d2b * gamma
    grad_coeff /= (0.001 + d2) * (w + gamma)
    return grad_coeff


def negumap_grad_coeff_rep(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    gamma = grad_args.gamma

    d2b = pow(d2, b)
    w = 1 / (1 + a * d2b)
    grad_coeff = 2.0 * a * b * w * w * d2b
    grad_coeff /= (0.001 + d2) * (w + gamma)
    return grad_coeff


class NegUMAP(CustomGradientUMAP2):
    def get_gradient_args(self):
        class UmapGradientArgs(NamedTuple):
            a: float
            b: float
            gamma: float

        return UmapGradientArgs(a=self._a, b=self._b, gamma=self.repulsion_strength)

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True

        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=negumap_grad_coeff_attr,
            custom_rep_func=negumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class NegUmap(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return run_embed(x, params, NegUMAP, "Neg-UMAP")
