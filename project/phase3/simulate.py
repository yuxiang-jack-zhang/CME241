"""Monte Carlo simulation engine and baseline strategies."""

import numpy as np
import torch
from .config import (
    T, BETA, W0, W_MIN,
    N_INCOME, INCOME_VALS, INCOME_TRANS,
    N_REGIME, REGIME_TRANS,
    RETURN_SCENARIOS, N_RETURN_SCENARIOS,
)
from .utils import crra_scalar
from .env import PortfolioEnv


def simulate_strategy(strategy_fn, n_paths=2000, seed=42):
    """Simulate wealth paths under a given strategy function.

    strategy_fn(t, W, y, z) -> (cons_frac, weights_array)

    Returns: wealth_paths (n, T+1), cons_paths (n, T), alloc_paths (n, T, 3), utilities (n,)
    """
    rng = np.random.RandomState(seed)
    wealth = np.full((n_paths, T + 1), W0)
    cons = np.zeros((n_paths, T))
    allocs = np.zeros((n_paths, T, 3))
    cum_util = np.zeros(n_paths)

    y_state = np.ones(n_paths, dtype=int)
    z_state = np.ones(n_paths, dtype=int)

    for t in range(T):
        for i in range(n_paths):
            w = wealth[i, t]
            yi, zi = y_state[i], z_state[i]

            cf, weights = strategy_fn(t, w, yi, zi)
            allocs[i, t] = weights
            c = cf * w
            cons[i, t] = c
            savings = w - c

            cum_util[i] += BETA ** t * crra_scalar(c)

            scen = RETURN_SCENARIOS[zi]
            ki = rng.choice(N_RETURN_SCENARIOS, p=scen["probs"])
            gross_ret = weights @ scen["returns"][ki]
            yp = rng.choice(N_INCOME, p=INCOME_TRANS[yi])
            zp = rng.choice(N_REGIME, p=REGIME_TRANS[zi])

            w_next = savings * gross_ret + INCOME_VALS[yp]
            wealth[i, t + 1] = np.clip(w_next, W_MIN, W0 * 8)  # W_MAX=800
            y_state[i] = yp
            z_state[i] = zp

    for i in range(n_paths):
        if wealth[i, -1] > 1e-10:
            cum_util[i] += BETA ** T * crra_scalar(wealth[i, -1])
        else:
            cum_util[i] += -1e12

    return wealth, cons, allocs, cum_util


# ── Strategy wrappers ──

def make_linear_q_strategy(agent):
    """Wrap a LinearQAgent into a strategy function."""
    def strategy(t, W, y, z):
        env_tmp = PortfolioEnv()
        env_tmp.t, env_tmp.W, env_tmp.y, env_tmp.z = t, W, y, z
        env_tmp.done = False
        state = env_tmp.featurize()
        q = agent.q_values(state)
        action = int(np.argmax(q))
        return env_tmp.decode_action(action)
    return strategy


def make_dqn_strategy(agent):
    """Wrap a DQNAgent into a strategy function."""
    from .agents.dqn import device
    def strategy(t, W, y, z):
        env_tmp = PortfolioEnv()
        env_tmp.t, env_tmp.W, env_tmp.y, env_tmp.z = t, W, y, z
        env_tmp.done = False
        state = env_tmp.featurize()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            q = agent.q_net(s)
            action = int(q.argmax(dim=1).item())
        return env_tmp.decode_action(action)
    return strategy


def make_ppo_strategy(agent):
    """Wrap a PPOAgent into a strategy function (greedy/deterministic)."""
    def strategy(t, W, y, z):
        env_tmp = PortfolioEnv()
        env_tmp.t, env_tmp.W, env_tmp.y, env_tmp.z = t, W, y, z
        env_tmp.done = False
        state = env_tmp.featurize()
        action = agent.greedy_action(state)
        return env_tmp.decode_action(action)
    return strategy


def make_strategy(agent):
    """Generic wrapper: detect agent type and return appropriate strategy function."""
    if hasattr(agent, 'actor'):
        return make_ppo_strategy(agent)
    elif hasattr(agent, 'q_net'):
        return make_dqn_strategy(agent)
    elif hasattr(agent, 'q_values'):
        return make_linear_q_strategy(agent)
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")


# ── Baseline strategies ──

def static_60_40(t, W, y, z):
    return 0.04, np.array([0.60, 0.40, 0.00])


def static_equal(t, W, y, z):
    return 0.04, np.array([1/3, 1/3, 1/3])


def glidepath_strategy(t, W, y, z):
    stock = 0.80 - 0.60 * t / max(T - 1, 1)
    bond = 1.0 - stock
    return 0.04, np.array([stock, bond, 0.0])
