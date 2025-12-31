from dataclasses import dataclass

import numba
import numpy as np
from numba.experimental import jitclass
from umap.layouts import clip, rdist, tau_rand_int

import drnb.embed.umap
from drnb.embed import fit_transform_embed
from drnb.embed.context import EmbedContext
from drnb.embed.umap.custom import CustomGradientUMAP
from drnb.types import EmbedResult


@jitclass([("z_bar", numba.float32), ("gamma", numba.float32)])
class NegTsneGradientArgs:
    """Parameters for Neg-t-SNE gradients."""

    def __init__(self, z_bar: float, gamma: float):
        self.z_bar = z_bar
        self.gamma = gamma


def negtsne_grad_coeff_attr(
    dist_squared: float, grad_args: NegTsneGradientArgs
) -> float:
    w = 1 / (1 + dist_squared)
    grad_coeff = -2.0 * grad_args.gamma * w
    grad_coeff /= w + grad_args.gamma
    return grad_coeff


def negtsne_grad_coeff_rep(
    dist_squared: float, grad_args: NegTsneGradientArgs
) -> float:
    w = 1 / (1 + dist_squared)
    grad_coeff = 2.0 * w * w
    grad_coeff /= w + grad_args.gamma
    return grad_coeff


def negtsne_epoch_func(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    rng_state,
    dim,
    move_other,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    alpha,
    grad_coeff_attr,
    grad_coeff_rep,
    grad_args,
):
    """Perform a single Neg-t-SNE epoch update."""
    z_bar_xi = 2.0 * grad_args.z_bar / (n_vertices * (n_vertices - 1))

    # pylint: disable=not-an-iterable
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] > n:
            continue
        n_neg_samples = int(
            (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
        )
        epoch_of_next_sample[i] += epochs_per_sample[i]
        epoch_of_next_negative_sample[i] += (
            n_neg_samples * epochs_per_negative_sample[i]
        )
        grad_args.gamma = z_bar_xi * n_neg_samples

        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]

        dist_squared = rdist(current, other)

        if dist_squared > 0.0:
            grad_coeff = grad_coeff_attr(dist_squared, grad_args)
        else:
            grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))

            current[d] += grad_d * alpha
            if move_other:
                other[d] += -grad_d * alpha

        for _ in range(n_neg_samples):
            k = tau_rand_int(rng_state) % n_vertices
            if j == k:
                continue

            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = grad_coeff_rep(dist_squared, grad_args)
            else:
                grad_coeff = 0.0

            for d in range(dim):
                if grad_coeff > 0.0:
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                else:
                    grad_d = 4.0
                current[d] += grad_d * alpha


class NegTSNE(CustomGradientUMAP):
    """Neg-t-SNE implementation."""

    def get_gradient_args(self):
        return NegTsneGradientArgs(z_bar=self.repulsion_strength, gamma=0.0)

    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = False
        super().__init__(
            custom_attr_func=negtsne_grad_coeff_attr,
            custom_rep_func=negtsne_grad_coeff_rep,
            custom_epoch_func=negtsne_epoch_func,
            **kwargs,
        )


@dataclass
class NegTsne(drnb.embed.umap.Umap):
    """Embedder for Neg-t-SNE."""

    use_precomputed_knn: bool = True
    drnb_init: str | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        return fit_transform_embed(x, params, NegTSNE, "Neg-t-SNE")
