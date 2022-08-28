import numpy as np


def center(X):
    return X - np.mean(X, axis=0)
