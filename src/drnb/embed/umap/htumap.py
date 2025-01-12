from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.types import EmbedResult


class HTUMAPGradientArgs(NamedTuple):
    """Gradient arguments for Heavy-Tailed UMAP."""

    a: float
    b_div_a: float
    a1a: float
    ma: float
    b2: float
    mba2: float


def htumap_grad_coeff_attr(d2: float, grad_args: HTUMAPGradientArgs) -> float:
    """Gradient coefficient for the attractive cost function in Heavy-Tailed UMAP."""
    a = grad_args.a
    mba2 = grad_args.mba2

    return mba2 / (a + d2)


def htumap_grad_coeff_rep(d2: float, grad_args: HTUMAPGradientArgs) -> float:
    """Gradient coefficient for the repulsive cost function in Heavy-Tailed UMAP."""
    ba = grad_args.b_div_a
    ma = grad_args.ma
    a1a = grad_args.a1a
    b2 = grad_args.b2

    w = pow(1.0 + ba * d2, ma)
    return b2 * pow(w, a1a) / (1.001 - w)


class HTUMAP(CustomGradientUMAP2):
    """Heavy-Tailed UMAP implementation."""

    def get_gradient_args(self):
        a = self._a
        b = self._b

        return HTUMAPGradientArgs(
            a=a,
            b_div_a=b / a,
            a1a=(a + 1.0) / a,
            ma=-1.0 * a,
            b2=b * 2.0,
            mba2=-2.0 * b * a,
        )

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True
        # if b is not explicitly defined, a will get over-written
        if "b" not in kwargs:
            kwargs["b"] = 1.0
        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=htumap_grad_coeff_attr,
            custom_rep_func=htumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Htumap(drnb.embed.umap.Umap):
    """Embedder for HT-UMAP."""

    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, HTUMAP, "HT-UMAP")


class HTNegUMAPGradientArgs(NamedTuple):
    """Gradient arguments for Heavy-Tailed NegUMAP."""

    a: float
    b: float
    b_div_a: float
    ainv: float
    a1a: float
    gamma: float


def htnegumap_grad_coeff_attr(d2: float, grad_args: HTNegUMAPGradientArgs) -> float:
    """Gradient coefficient for the attractive cost function in Heavy-Tailed NegUMAP."""
    a = grad_args.a
    b = grad_args.b
    ba = grad_args.b_div_a
    ainv = grad_args.ainv
    gamma = grad_args.gamma

    w = pow(1 + ba * d2, -a)
    grad_coeff = -2.0 * gamma * b * pow(w, ainv)
    grad_coeff /= gamma + w
    return grad_coeff


def htnegumap_grad_coeff_rep(d2: float, grad_args: HTNegUMAPGradientArgs) -> float:
    """Gradient coefficient for the repulsive cost function in Heavy-Tailed NegUMAP."""
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
    """Heavy-Tailed NegUMAP implementation."""

    def get_gradient_args(self):
        return HTNegUMAPGradientArgs(
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
    """Embedder for HT-NegUMAP."""

    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, HTNegUMAP, "HT-NegUMAP")
