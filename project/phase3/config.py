"""Shared constants and parameters for the lifecycle portfolio problem."""

import numpy as np

# ── Horizon & preferences ──
T = 40               # periods (years) to retirement
GAMMA = 2.0          # CRRA risk-aversion
BETA = 0.96          # per-period discount factor
W0 = 100.0           # initial wealth
W_MIN = 0.01         # floor for wealth (avoid log(0))
W_MAX = 800.0        # cap for wealth

# ── Income states: low / medium / high ──
N_INCOME = 3
INCOME_VALS = np.array([3.0, 10.0, 25.0])
INCOME_TRANS = np.array([
    [0.70, 0.25, 0.05],   # from low
    [0.15, 0.70, 0.15],   # from medium
    [0.05, 0.25, 0.70],   # from high
])
INCOME_LABELS = ["Low", "Medium", "High"]

# ── Market regimes: bear / normal / bull ──
N_REGIME = 3
REGIME_TRANS = np.array([
    [0.50, 0.40, 0.10],   # from bear
    [0.20, 0.60, 0.20],   # from normal
    [0.10, 0.40, 0.50],   # from bull
])
REGIME_LABELS = ["Bear", "Normal", "Bull"]

# Asset return scenarios per regime: (stocks_gross, bonds_gross, cash_gross)
# 3 scenarios per regime with probabilities
RETURN_SCENARIOS = {
    0: {  # bear
        "probs": np.array([0.4, 0.4, 0.2]),
        "returns": np.array([
            [0.82, 1.04, 1.005],   # bad
            [0.93, 1.03, 1.005],   # medium
            [1.02, 1.02, 1.005],   # good
        ]),
    },
    1: {  # normal
        "probs": np.array([0.25, 0.50, 0.25]),
        "returns": np.array([
            [0.92, 1.02, 1.005],
            [1.06, 1.03, 1.005],
            [1.14, 1.04, 1.005],
        ]),
    },
    2: {  # bull
        "probs": np.array([0.2, 0.4, 0.4]),
        "returns": np.array([
            [0.98, 1.01, 1.005],
            [1.12, 1.03, 1.005],
            [1.25, 1.05, 1.005],
        ]),
    },
}
N_RETURN_SCENARIOS = 3

# ── Action space ──
# Portfolio allocations: 20% increments across 3 assets
ALLOC_LIST = []
for s in range(6):           # stocks: 0.0, 0.2, ..., 1.0
    for b in range(6 - s):   # bonds:  0.0, 0.2, ..., 1.0 - s
        c = 5 - s - b        # cash = remainder
        ALLOC_LIST.append(np.array([s, b, c]) * 0.20)
ALLOC_ARRAY = np.array(ALLOC_LIST)  # shape (21, 3)
N_ALLOC = len(ALLOC_ARRAY)

# Consumption fractions
CONS_FRACS = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
N_CONS = len(CONS_FRACS)

N_ACTIONS = N_CONS * N_ALLOC  # 231
