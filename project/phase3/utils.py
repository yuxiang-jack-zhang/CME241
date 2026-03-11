"""CRRA utility functions."""

import numpy as np
from .config import GAMMA


def crra_utility(c, gamma=GAMMA):
    """CRRA utility; returns large negative value for c <= 0."""
    c = np.asarray(c, dtype=float)
    result = np.full_like(c, -1e12)
    mask = c > 1e-10
    if abs(gamma - 1.0) < 1e-8:
        result[mask] = np.log(c[mask])
    else:
        result[mask] = c[mask] ** (1.0 - gamma) / (1.0 - gamma)
    return result


def crra_scalar(c, gamma=GAMMA):
    """Scalar CRRA utility."""
    if c <= 1e-10:
        return -1e12
    if abs(gamma - 1.0) < 1e-8:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)
