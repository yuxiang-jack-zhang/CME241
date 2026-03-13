"""Lifecycle portfolio allocation environment with Gym-like interface."""

import numpy as np
from .config import (
    T, BETA, W0, W_MIN, W_MAX,
    N_INCOME, INCOME_VALS, INCOME_TRANS,
    N_REGIME, REGIME_TRANS,
    RETURN_SCENARIOS, N_RETURN_SCENARIOS,
    ALLOC_ARRAY, CONS_FRACS, N_ALLOC,
)
from .utils import crra_scalar


class PortfolioEnv:
    """Lifecycle portfolio allocation environment."""

    def __init__(self):
        self.t = 0
        self.W = W0
        self.y = 1     # start medium income
        self.z = 1     # start normal regime
        self.done = False

    def reset(self, W0_val=None, y0=None, z0=None, randomize=False):
        """Reset to initial state. Returns feature vector.

        If randomize=True, sample diverse starting conditions so the agent
        experiences a wider range of (wealth, income, regime) states early
        in training, which is critical for learning state-dependent policies.
        """
        self.t = 0
        if randomize:
            self.W = np.exp(np.random.uniform(np.log(10), np.log(W_MAX * 0.6)))
            self.y = np.random.randint(N_INCOME)
            self.z = np.random.randint(N_REGIME)
        else:
            self.W = W0 if W0_val is None else W0_val
            self.y = 1 if y0 is None else y0
            self.z = 1 if z0 is None else z0
        self.done = False
        return self.featurize()

    def featurize(self):
        """Convert state to fixed-length feature vector (dim=8)."""
        t_feat = self.t / T
        w_feat = np.log(max(self.W, W_MIN)) / np.log(W_MAX)
        y_onehot = np.zeros(N_INCOME)
        y_onehot[self.y] = 1.0
        z_onehot = np.zeros(N_REGIME)
        z_onehot[self.z] = 1.0
        return np.concatenate([[t_feat, w_feat], y_onehot, z_onehot])

    @property
    def state_dim(self):
        return 2 + N_INCOME + N_REGIME   # 8

    def decode_action(self, action_idx):
        """Map action index to (cons_frac, allocation_weights)."""
        ci = action_idx // N_ALLOC
        ai = action_idx % N_ALLOC
        return CONS_FRACS[ci], ALLOC_ARRAY[ai]

    def step(self, action_idx):
        """Take one step. Returns (next_features, reward, done, info)."""
        assert not self.done, "Episode is done, call reset()"

        cons_frac, weights = self.decode_action(action_idx)

        # Consumption
        consumption = cons_frac * self.W
        savings = self.W - consumption

        # Discounted CRRA utility of consumption (β^t accounts for time
        # value of money). Agents use gamma=1.0 since discounting is here.
        reward = BETA ** self.t * crra_scalar(consumption)

        # Transition: sample return scenario
        scen = RETURN_SCENARIOS[self.z]
        ki = np.random.choice(N_RETURN_SCENARIOS, p=scen["probs"])
        gross_returns = scen["returns"][ki]
        portfolio_return = weights @ gross_returns

        # Transition: income and regime
        yp = np.random.choice(N_INCOME, p=INCOME_TRANS[self.y])
        zp = np.random.choice(N_REGIME, p=REGIME_TRANS[self.z])

        # Next wealth
        W_next = savings * portfolio_return + INCOME_VALS[yp]
        W_next = np.clip(W_next, W_MIN, W_MAX)

        self.t += 1
        self.W = W_next
        self.y = yp
        self.z = zp

        # Check terminal
        if self.t >= T:
            self.done = True
            reward += BETA ** T * crra_scalar(self.W)

        info = {"consumption": consumption, "savings": savings,
                "portfolio_return": portfolio_return, "W": self.W}

        return self.featurize(), reward, self.done, info
