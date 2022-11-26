from dataclasses import dataclass
from typing import NamedTuple

import drnb.embed.umap
from drnb.embed.umap.custom2 import CustomGradientUMAP2, epoch_func
from drnb.log import log


def pacumap_grad_coeff_attr(d2, grad_args):
    a = grad_args.a
    b = grad_args.b

    w = 1 / (1 + d2)
    bw1 = 1 + b * w
    grad_coeff = -2.0 * a * b * w * w
    grad_coeff /= bw1 * bw1
    return grad_coeff


# pylint: disable=unused-argument
def pacumap_grad_coeff_rep(d2, grad_args):
    w = 1 / (1 + d2)
    w1 = 1 + w
    grad_coeff = 2.0 * w * w
    grad_coeff /= w1 * w1
    return grad_coeff


class PacUMAP(CustomGradientUMAP2):
    def get_gradient_args(self):
        class GradientArgs(NamedTuple):
            a: float
            b: float

        return GradientArgs(a=self._a, b=self._b)

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
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return embed_pacumap(x, params)


def embed_pacumap(
    x,
    params,
):
    log.info("Running PacUMAP")
    embedder = PacUMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
