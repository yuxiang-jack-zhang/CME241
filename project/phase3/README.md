# Phase 3: Lifecycle Portfolio Allocation via RL

Modular Python package for training and evaluating RL agents on a lifecycle portfolio allocation problem with 3 market regimes, 3 income states, and continuous wealth.

## Directory Structure

```
phase3/
├── config.py          # All shared constants & parameters
├── utils.py           # CRRA utility functions
├── env.py             # PortfolioEnv (Gym-like interface)
├── agents/
│   ├── base.py        # Abstract BaseAgent interface
│   ├── linear_q.py    # Linear Q-learning agent
│   ├── dqn.py         # DQN agent (PyTorch)
│   └── __init__.py    # Agent registry
├── train.py           # Agent-agnostic training loop
├── simulate.py        # Monte Carlo simulation + baseline strategies
├── visualize.py       # All plotting and reporting functions
├── run.py             # Main entry point
└── plots/             # Generated plots (gitignored)
```

## Running

```bash
# From the repo root:
conda run -n cs124 python -m project.phase3.run

# Or directly:
conda run -n cs124 python project/phase3/run.py
```

This trains all registered agents, runs Monte Carlo simulations against baselines, prints comparison tables, and saves plots to `phase3/plots/`.

## Adding a New Agent

1. Create `agents/my_algo.py`:

```python
from .base import BaseAgent
import numpy as np

class MyAlgoAgent(BaseAgent):
    def __init__(self, state_dim, n_actions, **kwargs):
        # your init here
        self.n_actions = n_actions
        self.epsilon = 1.0

    @property
    def name(self) -> str:
        return "My Algo"

    def select_action(self, state) -> int:
        # your policy here
        return np.random.randint(self.n_actions)

    def update(self, state, action, reward, next_state, done):
        # online update (optional, for non-replay agents)
        pass

    def decay_epsilon(self, episode):
        self.epsilon = max(0.05, 1.0 - episode / 2000)
```

2. Register it in `agents/__init__.py`:

```python
from .my_algo import MyAlgoAgent

AGENT_REGISTRY = {
    "linear_q": LinearQAgent,
    "dqn": DQNAgent,
    "my_algo": MyAlgoAgent,   # <-- add this
}
```

3. Add training config in `run.py`:

```python
AGENT_CONFIGS["my_algo"] = dict(state_dim=8, n_actions=N_ACTIONS, ...)
TRAIN_EPISODES["my_algo"] = 5000
```

4. Run `python -m project.phase3.run` — your agent is automatically trained, simulated, and compared.

## Key Interfaces

- **`BaseAgent`** (`agents/base.py`): Abstract class all agents subclass. Required methods: `select_action(state)`, `decay_epsilon(episode)`, `name` property. Optional: `update()` for online agents, `store()` + `train_step()` for replay-based agents.

- **`PortfolioEnv`** (`env.py`): `reset() -> features`, `step(action_idx) -> (features, reward, done, info)`, `featurize() -> np.array(8,)`, `decode_action(idx) -> (cons_frac, weights)`.

- **`train_agent(agent, env, n_episodes)`** (`train.py`): Handles both online and replay-based agents automatically.

- **`simulate_strategy(fn, n_paths, seed)`** (`simulate.py`): Takes a function `fn(t, W, y, z) -> (cons_frac, weights)`. Use `make_strategy(agent)` to wrap any agent.

## State & Action Space

- **State features** (dim=8): `[t/T, log(W)/log(W_max), one_hot_income(3), one_hot_regime(3)]`
- **Actions** (231): 11 consumption fractions (0-20%) x 21 portfolio allocations (20% increments across stocks/bonds/cash)
