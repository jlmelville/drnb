from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.types import EmbedResult


class NegTUMAPGradientArgs(NamedTuple):
    """Gradient arguments for Neg-t-UMAP. Like NegUMAP but with a = 1, b = 1,
    thus simplifying the gradient calculations.

    Attributes:
        gamma: float: The repulsion strength as in UMAP.
    """

    gamma: float


def negtumap_grad_coeff_attr(d2: float, grad_args: NegTUMAPGradientArgs) -> float:
    """Gradient coefficient for the attractive cost function in Neg-t-UMAP."""
    gamma = grad_args.gamma

    w = 1 / (1 + d2)
    grad_coeff = -2.0 * gamma * w
    grad_coeff /= w + gamma
    return grad_coeff


def negtumap_grad_coeff_rep(d2: float, grad_args: NegTUMAPGradientArgs) -> float:
    """Gradient coefficient for the repulsive cost function in Neg-t-UMAP."""
    gamma = grad_args.gamma

    w = 1 / (1 + d2)
    grad_coeff = 2.0 * w * w
    grad_coeff /= w + gamma
    return grad_coeff


class NegTUMAP(CustomGradientUMAP2):
    """Neg-t-UMAP implementation."""

    def get_gradient_args(self):
        return NegTUMAPGradientArgs(gamma=self.repulsion_strength)

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
    """Embedder for Neg-t-UMAP."""

    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, NegTUMAP, "Neg-t-UMAP")
