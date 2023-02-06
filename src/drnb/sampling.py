import numpy as np


# sampling in the ncvis style of slowly increasing the number of negatives over
# epochs (but on average sticking to `n_negative`)
def ncvis_negative_plan(n_negative=5, n_epochs=200):
    negative_plan = np.linspace(0, 1, n_epochs)
    negative_plan = negative_plan**3

    negative_plan /= negative_plan.sum()
    negative_plan *= n_epochs * n_negative
    negative_plan = negative_plan.round().astype(np.int)
    negative_plan[negative_plan < 1] = 1
    return negative_plan


def create_sample_plan(n_samples, n_epochs, strategy="unif"):
    if strategy == "inc":
        samples = ncvis_negative_plan(n_samples, n_epochs)
    elif strategy == "dec":
        samples = np.flip(ncvis_negative_plan(n_samples, n_epochs))
    elif strategy == "unif":
        samples = np.array([n_samples] * n_epochs, dtype=np.int)
    else:
        raise ValueError(f"Unknown sample strategy {strategy}")
    return samples
