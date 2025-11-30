from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.types import EmbedResult


class NegUmapGradientArgs(NamedTuple):
    """Gradient arguments for NegUMAP.

    Attributes:
        a: float: The a parameter as in UMAP.
        b: float: The b parameter as in UMAP.
        gamma: float: The repulsion strength as in UMAP.
    """

    a: float
    b: float
    gamma: float


def negumap_grad_coeff_attr(d2: float, grad_args: NegUmapGradientArgs) -> float:
    """Gradient coefficient for the attractive cost function in NegUMAP."""
    a = grad_args.a
    b = grad_args.b
    gamma = grad_args.gamma

    d2b = pow(d2, b)
    w = 1 / (1 + a * d2b)
    grad_coeff = -2.0 * a * b * w * d2b * gamma
    grad_coeff /= (0.001 + d2) * (w + gamma)
    return grad_coeff


def negumap_grad_coeff_rep(d2: float, grad_args: NegUmapGradientArgs) -> float:
    """Gradient coefficient for the repulsive cost function in NegUMAP."""
    a = grad_args.a
    b = grad_args.b
    gamma = grad_args.gamma

    d2b = pow(d2, b)
    w = 1 / (1 + a * d2b)
    grad_coeff = 2.0 * a * b * w * w * d2b
    grad_coeff /= (0.001 + d2) * (w + gamma)
    return grad_coeff


class NegUMAP(CustomGradientUMAP2):
    """NegUMAP implementation."""

    def get_gradient_args(self):
        return NegUmapGradientArgs(a=self._a, b=self._b, gamma=self.repulsion_strength)

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
    """Embedder for NegUMAP."""

    use_precomputed_knn: bool = True
    drnb_init: str | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, NegUMAP, "Neg-UMAP")
