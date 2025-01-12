from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.types import EmbedResult


class PacUMAPGradientArgs(NamedTuple):
    """Gradient arguments for PacUMAP.

    Cost function for attraction is a / (1 + bw) where (w = 1 / (1 + d2)).

    Attributes:
        a: float: The a parameter in the attractive cost function.
        b: float: The b parameter in the attractive cost function.
    """

    a: float
    b: float


def pacumap_grad_coeff_attr(d2: float, grad_args: PacUMAPGradientArgs) -> float:
    """Gradient coefficient for the attractive cost function in PacUMAP."""
    a = grad_args.a
    b = grad_args.b

    w = 1 / (1 + d2)
    bw1 = 1 + b * w
    grad_coeff = -2.0 * a * b * w * w
    grad_coeff /= bw1 * bw1
    return grad_coeff


def pacumap_grad_coeff_rep(d2: float, _) -> float:
    """Gradient coefficient for the repulsive cost function in PacUMAP."""
    w = 1 / (1 + d2)
    w1 = 1 + w
    grad_coeff = 2.0 * w * w
    grad_coeff /= w1 * w1
    return grad_coeff


class PacUMAP(CustomGradientUMAP2):
    """PacUMAP implementation."""

    def get_gradient_args(self):
        return PacUMAPGradientArgs(a=self._a, b=self._b)

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = True
        if "a" not in kwargs:
            kwargs["a"] = 1.0
        if "b" not in kwargs:
            kwargs["b"] = 10.0
        super().__init__(
            custom_epoch_func=epoch_func,
            custom_attr_func=pacumap_grad_coeff_attr,
            custom_rep_func=pacumap_grad_coeff_rep,
            **kwargs,
        )


@dataclass
class Pacumap(drnb.embed.umap.Umap):
    """Embedder for PacUMAP."""

    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, PacUMAP, "PacUMAP")
