"""DP baseline adapter for Phase 3 comparison.

Solves a simplified DP version of the lifecycle problem (no income/regime states)
using Phase 3's parameters, then wraps the policy as a strategy function for
simulation comparison. This gives us the optimality gap measurement.
"""

import numpy as np
from .config import (
    T, GAMMA, BETA, W0, W_MIN, W_MAX,
    CONS_FRACS, ALLOC_ARRAY, N_ALLOC,
    RETURN_SCENARIOS, N_RETURN_SCENARIOS,
    N_REGIME, REGIME_TRANS,
)
from .utils import crra_scalar


# Wealth grid for DP
N_WEALTH = 60
WEALTH_GRID = np.linspace(W_MIN, W_MAX, N_WEALTH)


def _crra_vec(c):
    c = np.asarray(c, dtype=float)
    result = np.full_like(c, -1e12)
    mask = c > 1e-10
    if abs(GAMMA - 1.0) < 1e-8:
        result[mask] = np.log(c[mask])
    else:
        result[mask] = c[mask] ** (1 - GAMMA) / (1 - GAMMA)
    return result


def _interp(v_arr, w_arr):
    """Linear interpolation on wealth grid."""
    w = np.clip(np.asarray(w_arr, dtype=float), WEALTH_GRID[0], WEALTH_GRID[-1])
    idx = np.searchsorted(WEALTH_GRID, w) - 1
    idx = np.clip(idx, 0, N_WEALTH - 2)
    frac = (w - WEALTH_GRID[idx]) / (WEALTH_GRID[idx + 1] - WEALTH_GRID[idx])
    return (1 - frac) * v_arr[idx] + frac * v_arr[idx + 1]


def _expected_return_scenarios():
    """Compute expected gross returns across all regimes (stationary distribution)."""
    # Use stationary distribution of regime Markov chain
    # Solve pi @ P = pi, sum(pi) = 1
    P = REGIME_TRANS
    A = np.vstack([P.T - np.eye(N_REGIME), np.ones(N_REGIME)])
    b = np.zeros(N_REGIME + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Weighted average across regimes and scenarios
    all_probs = []
    all_returns = []
    for z in range(N_REGIME):
        scen = RETURN_SCENARIOS[z]
        for k in range(N_RETURN_SCENARIOS):
            all_probs.append(pi[z] * scen["probs"][k])
            all_returns.append(scen["returns"][k])

    return np.array(all_probs), np.array(all_returns)


def solve_dp():
    """Backward induction DP on (t, W) grid.

    Uses expected returns averaged over the stationary regime distribution
    and medium income (deterministic) as a simplification.

    Returns:
        policy_cons:  (T, N_WEALTH) index into CONS_FRACS
        policy_alloc: (T, N_WEALTH) index into ALLOC_ARRAY
    """
    scenario_probs, scenario_returns = _expected_return_scenarios()
    n_scenarios = len(scenario_probs)
    mean_income = 10.0  # medium income

    # Precompute portfolio gross returns: (N_ALLOC, n_scenarios)
    port_gross = np.array([
        ALLOC_ARRAY[ai] @ scenario_returns[k]
        for ai in range(N_ALLOC)
        for k in range(n_scenarios)
    ]).reshape(N_ALLOC, n_scenarios)

    V = np.full((T + 1, N_WEALTH), -1e20)
    policy_cons = np.zeros((T, N_WEALTH), dtype=int)
    policy_alloc = np.zeros((T, N_WEALTH), dtype=int)

    # Terminal: utility of retirement wealth
    V[T] = _crra_vec(WEALTH_GRID)

    n_cons = len(CONS_FRACS)

    for t in range(T - 1, -1, -1):
        # consumptions: (n_cons, N_WEALTH)
        consumptions = CONS_FRACS[:, None] * WEALTH_GRID[None, :]
        remainings = WEALTH_GRID[None, :] - consumptions

        rewards = _crra_vec(consumptions)  # (n_cons, N_WEALTH)

        # next_w: (n_cons, N_WEALTH, N_ALLOC, n_scenarios)
        next_w = (remainings[:, :, None, None]
                  * port_gross[None, None, :, :]
                  + mean_income)

        future_v = _interp(
            V[t + 1], next_w.ravel()
        ).reshape(n_cons, N_WEALTH, N_ALLOC, n_scenarios)

        # expected future value
        ev = (future_v * scenario_probs[None, None, None, :]).sum(axis=3)

        # total value
        total = rewards[:, :, None] + BETA * ev
        total = np.where(remainings[:, :, None] <= 1e-10, -1e20, total)

        # best action per wealth point
        total_by_w = total.transpose(1, 0, 2).reshape(N_WEALTH, n_cons * N_ALLOC)
        best_flat = np.argmax(total_by_w, axis=1)

        V[t] = total_by_w[np.arange(N_WEALTH), best_flat]
        policy_cons[t] = best_flat // N_ALLOC
        policy_alloc[t] = best_flat % N_ALLOC

    return V, policy_cons, policy_alloc


def make_dp_strategy():
    """Solve DP and return a strategy function compatible with simulate_strategy."""
    print("  Solving DP baseline (backward induction)...")
    V, policy_cons, policy_alloc = solve_dp()
    print("  DP baseline solved.")

    def strategy(t, W, y, z):
        # Find nearest wealth grid point
        wi = np.argmin(np.abs(WEALTH_GRID - W))
        t_clamped = min(t, T - 1)
        ci = policy_cons[t_clamped, wi]
        ai = policy_alloc[t_clamped, wi]
        return CONS_FRACS[ci], ALLOC_ARRAY[ai]

    return strategy
