from typing import Protocol

import numba
import numpy as np
from numba.experimental import jitclass


# pylint: disable=too-few-public-methods
class OptimizerProtocol(Protocol):
    """Optimizer protocol."""

    # pylint: disable=missing-function-docstring
    def opt(
        self,
        Y: np.ndarray,
        grads: np.ndarray,
        epoch: int,
        n_epochs: int,
    ) -> np.ndarray: ...


def create_opt(
    nobs: int, opt: str | OptimizerProtocol, optargs=None
) -> OptimizerProtocol:
    """Create optimizer object by name."""
    # allow creation of optimizer object directly
    if not isinstance(opt, str):
        return opt

    if optargs is None:
        optargs = {}

    optargs["nobs"] = nobs
    optargs["ndim"] = 2

    if opt == "steepd":
        optim = SteepestDescent
    elif opt == "mom":
        optim = Momentum
    elif opt == "momgh":
        optim = MomentumGH
    elif opt == "adam":
        optim = Adam
    elif opt == "adabelief":
        optim = AdaBelief
    else:
        raise ValueError(f"Unknown optimizer {opt}")
    return optim(**optargs)


@numba.jit(nopython=True, fastmath=True)
def next_alpha(alpha, epoch, n_epochs):
    """Decay learning rate."""
    return alpha * (
        numba.float32(1.0) - (numba.float32(epoch) / numba.float32(n_epochs))
    )


@numba.jit(nopython=True, fastmath=True)
def warmup(alpha, epoch, n_warmup_epochs):
    """Linear warmup of learning rate."""
    if epoch >= n_warmup_epochs:
        return alpha
    return alpha * numba.float32(epoch + 1) / numba.float32(n_warmup_epochs)


@jitclass(
    [
        ("nobs", numba.int32),
        ("ndim", numba.int32),
        ("alpha", numba.float32),
        ("decay_alpha", numba.boolean),
    ]
)
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class SteepestDescent:
    """Steepest descent optimizer."""

    # pylint: disable=unused-argument
    def __init__(self, nobs, ndim, alpha=1.0, decay_alpha=True):
        self.alpha = alpha
        self.decay_alpha = decay_alpha

    def opt(
        self,
        Y: np.ndarray,
        grads: np.ndarray,
        epoch: int,
        n_epochs: int,
    ) -> np.ndarray:
        """Update parameters using optimizer."""
        if self.decay_alpha:
            alpha = next_alpha(self.alpha, epoch, n_epochs)
        else:
            alpha = self.alpha
        return Y - alpha * grads


@jitclass(
    [
        ("nobs", numba.int32),
        ("ndim", numba.int32),
        ("alpha", numba.float32),
        ("decay_alpha", numba.boolean),
        ("beta", numba.float32),
        ("memory", numba.float32[:, :]),
    ]
)
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class Momentum:
    """Momentum optimizer."""

    def __init__(self, nobs, ndim, alpha=1.0, decay_alpha=True, beta=0.5):
        self.alpha = alpha
        self.decay_alpha = decay_alpha
        self.beta = beta
        self.memory = np.zeros((nobs, ndim), dtype=np.float32)

    def opt(
        self, Y: np.ndarray, grads: np.ndarray, epoch: int, n_epochs: int
    ) -> np.ndarray:
        """Update parameters using optimizer."""
        if self.decay_alpha:
            alpha = next_alpha(self.alpha, epoch, n_epochs)
        else:
            alpha = self.alpha

        self.memory = self.beta * self.memory - alpha * grads

        return Y + self.memory


@jitclass(
    [
        ("nobs", numba.int32),
        ("ndim", numba.int32),
        ("alpha", numba.float32),
        ("decay_alpha", numba.boolean),
        ("beta1", numba.float32),
        ("memory", numba.float32[:, :]),
    ]
)
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class MomentumGH:
    """Momentum optimizer as a weighted sum of previous gradients. Decouples direction
    from step size."""

    def __init__(self, nobs, ndim, alpha=1.0, decay_alpha=True, beta=0.5):
        self.alpha = alpha
        self.decay_alpha = decay_alpha
        self.beta1 = numba.float32(1.0 - beta)
        self.memory = np.zeros((nobs, ndim), dtype=np.float32)

    def opt(
        self, Y: np.ndarray, grads: np.ndarray, epoch: int, n_epochs: int
    ) -> np.ndarray:
        """Update parameters using optimizer."""
        if self.decay_alpha:
            alpha = next_alpha(self.alpha, epoch, n_epochs)
        else:
            alpha = self.alpha

        # self.memory = self.beta * self.memory + self.beta1 * grads
        self.memory += self.beta1 * (grads - self.memory)
        return Y - alpha * self.memory


@jitclass(
    [
        ("nobs", numba.int32),
        ("ndim", numba.int32),
        ("alpha", numba.float32),
        ("decay_alpha", numba.boolean),
        ("beta1", numba.float32),
        ("beta2", numba.float32),
        ("beta11", numba.float32),
        ("beta21", numba.float32),
        ("eps", numba.float32),
        ("m", numba.float32[:, :]),
        ("v", numba.float32[:, :]),
        ("one", numba.float32),
        ("warmup_alpha", numba.int32),
    ]
)
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class Adam:
    """Adam optimizer."""

    def __init__(
        self,
        nobs,
        ndim,
        alpha=1e-3,
        decay_alpha=False,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        warmup_alpha=0,
    ):
        self.one = numba.float32(1.0)
        self.alpha = alpha
        self.decay_alpha = decay_alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta11 = self.one - beta1
        self.beta21 = self.one - beta2
        self.eps = eps
        self.warmup_alpha = warmup_alpha
        self.m = np.zeros((nobs, ndim), dtype=np.float32)
        self.v = np.zeros((nobs, ndim), dtype=np.float32)

    def opt(
        self, Y: np.ndarray, grads: np.ndarray, epoch: int, n_epochs: int
    ) -> np.ndarray:
        """Update parameters using optimizer."""
        e1 = numba.float32(epoch + 1)

        if self.decay_alpha:
            alpha = next_alpha(self.alpha, epoch, n_epochs)
        else:
            alpha = self.alpha
        alpha = warmup(alpha, epoch, self.warmup_alpha)
        vbcorr = np.sqrt(self.one - self.beta2**e1)
        alpha *= vbcorr / (self.one - self.beta1**e1)

        self.m = self.beta1 * self.m + self.beta11 * grads
        self.v = self.beta2 * self.v + self.beta21 * grads * grads

        return Y - alpha * (self.m / (np.sqrt(self.v) + self.eps / vbcorr))


@jitclass(
    [
        ("nobs", numba.int32),
        ("ndim", numba.int32),
        ("alpha", numba.float32),
        ("decay_alpha", numba.boolean),
        ("beta1", numba.float32),
        ("beta2", numba.float32),
        ("beta11", numba.float32),
        ("beta21", numba.float32),
        ("eps", numba.float32),
        ("m", numba.float32[:, :]),
        ("v", numba.float32[:, :]),
        ("one", numba.float32),
        ("warmup_alpha", numba.int32),
    ]
)
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class AdaBelief:
    """AdaBelief optimizer."""

    def __init__(
        self,
        nobs,
        ndim,
        alpha=1e-3,
        decay_alpha=False,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        warmup_alpha=0,
    ):
        self.one = numba.float32(1.0)
        self.alpha = alpha
        self.decay_alpha = decay_alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta11 = self.one - beta1
        self.beta21 = self.one - beta2
        self.eps = eps
        self.warmup_alpha = warmup_alpha
        self.m = np.zeros((nobs, ndim), dtype=np.float32)
        self.v = np.zeros((nobs, ndim), dtype=np.float32)

    def opt(
        self, Y: np.ndarray, grads: np.ndarray, epoch: int, n_epochs: int
    ) -> np.ndarray:
        """Update parameters using optimizer."""
        e1 = numba.float32(epoch + 1)

        if self.decay_alpha:
            alpha = next_alpha(self.alpha, epoch, n_epochs)
        else:
            alpha = self.alpha
        alpha = warmup(alpha, epoch, self.warmup_alpha)
        vbcorr = np.sqrt(self.one - self.beta2**e1)
        alpha *= vbcorr / (self.one - self.beta1**e1)

        # self.m = self.beta1 * self.m + self.beta11 * grads
        self.m += self.beta11 * (grads - self.m)
        # self.v = self.beta2 * self.v + self.beta21 * (grads - self.m)**2 + eps
        self.v += self.beta21 * ((grads - self.m) ** 2 - self.v) + self.eps

        return Y - alpha * (self.m / (np.sqrt(self.v) + self.eps / vbcorr))
