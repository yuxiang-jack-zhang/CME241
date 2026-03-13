"""CRRA utility functions and training helpers."""

import numpy as np
from .config import GAMMA


class RewardNormalizer:
    """Running mean/std reward normalizer for neural network agents."""

    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward):
        if self.count < 2:
            return reward
        std = max(np.sqrt(self.var / max(self.count - 1, 1)), 1e-8)
        return (reward - self.mean) / std


def crra_utility(c, gamma=GAMMA):
    """CRRA utility; returns large negative value for c <= 0.

    Penalty is -1e4 (RL-friendly) rather than -inf, but still far worse
    than the worst legitimate utility (~-5000 at minimum wealth).
    """
    c = np.asarray(c, dtype=float)
    result = np.full_like(c, -1e4)
    mask = c > 1e-10
    if abs(gamma - 1.0) < 1e-8:
        result[mask] = np.log(c[mask])
    else:
        result[mask] = c[mask] ** (1.0 - gamma) / (1.0 - gamma)
    return result


def crra_scalar(c, gamma=GAMMA):
    """Scalar CRRA utility."""
    if c <= 1e-10:
        return -1e4
    if abs(gamma - 1.0) < 1e-8:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)
