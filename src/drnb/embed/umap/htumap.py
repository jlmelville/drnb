from dataclasses import dataclass
from typing import NamedTuple

import drnb.embed.umap
from drnb.embed import run_embed
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func


def htumap_grad_coeff_attr(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    ba = grad_args.b_div_a
    ainv = grad_args.ainv

    w = pow(1 + ba * d2, -a)
    return -2.0 * b * pow(w, ainv)


def htumap_grad_coeff_rep(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    ba = grad_args.b_div_a
    a1a = grad_args.a1a

    w = pow(1 + ba * d2, -a)
    grad_coeff = 2.0 * b * pow(w, a1a)
    return grad_coeff / (1.001 - w)


class HTUMAP(CustomGradientUMAP2):
    def get_gradient_args(self):
        class GradientArgs(NamedTuple):
            a: float
            b: float
            b_div_a: float
            ainv: float
            a1a: float

        return GradientArgs(
            a=self._a,
            b=self._b,
            b_div_a=self._b / self._a,
            ainv=1 / self._a,
            a1a=(self._a + 1) / self._a,
        )

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True
        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=htumap_grad_coeff_attr,
            custom_rep_func=htumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Htumap(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return run_embed(x, params, HTUMAP, "HT-UMAP")


def htnegumap_grad_coeff_attr(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    ba = grad_args.b_div_a
    ainv = grad_args.ainv
    gamma = grad_args.gamma

    w = pow(1 + ba * d2, -a)
    grad_coeff = -2.0 * gamma * b * pow(w, ainv)
    grad_coeff /= gamma + w
    return grad_coeff


def htnegumap_grad_coeff_rep(d2, grad_args):
    a = grad_args.a
    b = grad_args.b
    ba = grad_args.b_div_a
    a1a = grad_args.a1a
    gamma = grad_args.gamma

    w = pow(1 + ba * d2, -a)
    grad_coeff = 2.0 * b * pow(w, a1a)
    grad_coeff /= gamma + w
    return grad_coeff


class HTNegUMAP(CustomGradientUMAP2):
    def get_gradient_args(self):
        class GradientArgs(NamedTuple):
            a: float
            b: float
            b_div_a: float
            ainv: float
            a1a: float
            gamma: float

        return GradientArgs(
            a=self._a,
            b=self._b,
            b_div_a=self._b / self._a,
            ainv=1 / self._a,
            a1a=(self._a + 1) / self._a,
            gamma=self.repulsion_strength,
        )

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True
        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=htnegumap_grad_coeff_attr,
            custom_rep_func=htnegumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Htnegumap(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return run_embed(x, params, HTNegUMAP, "HT-NegUMAP")
