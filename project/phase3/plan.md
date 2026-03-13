# Phase 3 Implementation Plan

Based on critique.md analysis and discussion. Priorities ordered by impact.

---

## 1. Ensure All Models Appear in All Plots

**Goal:** Every graph (allocation_stacks, consumption_paths, learning_curves, terminal_wealth, utility_distributions, wealth_trajectories) must show all 4 RL agents (Linear Q, SARSA, DQN, PPO) + all 3 baselines (Static 60/40, Equal 1/3, Glidepath).

**Changes:**
- Verify SARSA is included in AGENT_REGISTRY and appears in all plots (currently trains only 4,000 episodes vs 15,000 — bump to at least 10,000)
- Verify `plot_consumption_paths` n_rl_agents count is correct when all 4 agents are present
- Confirm all 7 strategies appear in every simulation plot (not just some)

---

## 2. Auto-Generated report.md

**Goal:** Create a `report.md` that summarizes setup, models, and results, regenerated on every run.

**Implementation:**
- Add a `generate_report()` function (in `visualize.py` or new `report.py`) that writes markdown containing:
  - Environment setup: T, GAMMA, BETA, W0, action space size (N_ACTIONS), state dim, income/regime parameters
  - Agent descriptions: name, type (linear/neural), key hyperparameters, number of training episodes
  - Comparison table: E[Utility], E[W_T], Med[W_T], Std[W_T], P10, P90, P(ruin) — same as print_comparison_table but in markdown
  - Risk metrics table: MaxDrawdown, Sharpe(C), MinW_T — same as print_risk_metrics but in markdown
  - Embedded image references to all plots in `plots/`
- Call `generate_report()` at the end of `run.py` main() after all plots are saved
- The implementation agent must remember to call this function whenever results are regenerated

---

## 3. Fix DQN Degenerate Policy

DQN currently outputs a uniform policy (100% stocks, 20% consumption everywhere). Root causes and fixes, in priority order:

### 3A. Double DQN (do first — trivial 2-line change)

Standard DQN overestimates Q-values. With 231 actions, the max over 231 noisy Q-values is severely biased upward, making all actions look equally "good."

**Change in `agents/dqn.py` train_step():**
```python
# BEFORE (line ~122):
q_next = self.target_net(ns).max(dim=1)[0]

# AFTER (Double DQN):
best_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
q_next = self.target_net(ns).gather(1, best_actions).squeeze(1)
```

### 3B. Reward Normalization

CRRA rewards range from -1e4 (zero consumption) to ~-0.01 (high consumption), spanning orders of magnitude. Neural nets can't fit this range well. Linear Q handles it because linear features naturally scale.

**Implementation:** Add a running reward normalizer class that tracks mean/std of observed rewards and normalizes before feeding to the network. Apply to both DQN and PPO.

```python
class RewardNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward):
        std = max(np.sqrt(self.var / max(self.count, 1)), 1e-8)
        return (reward - self.mean) / std
```

### 3C. Larger Replay Buffer + Slower Target Updates

With 231 actions × diverse states, 100k buffer may not have enough coverage. Frequent target updates (every 200 steps) cause instability.

**Changes in run.py AGENT_CONFIGS["dqn"]:**
- `buffer_size`: 100000 → 500000
- `target_update_freq`: 200 → 1000

### 3D. Learning Rate Scheduling

The network may oscillate around a flat policy because LR is too high for fine-tuning after initial learning.

**Implementation:** Add cosine or step LR decay from 3e-4 → 1e-5 over training. Use `torch.optim.lr_scheduler.CosineAnnealingLR` in the DQN agent, stepped each episode.

---

## 4. Fix PPO Degenerate Policy

PPO shows ~20% stocks / ~20% consumption almost everywhere. Similar root causes to DQN but PPO-specific fixes:

### 4A. Reward Normalization (same as 3B)

PPO's value head tries to predict cumulative reward. If the scale is wrong, advantages are noisy and the policy doesn't improve. Apply the same RewardNormalizer from 3B.

### 4B. Value Function Clipping

Add value function clipping to prevent the critic from overshooting, which destabilizes the advantage estimates.

### 4C. Entropy Coefficient Tuning

Current ent_coef=0.05 is relatively high. If the policy is too uniform, entropy bonus may be preventing it from committing to good actions. Consider decaying entropy from 0.05 → 0.01 over training.

---

## 5. Add DP Baseline from Phase 2 (Missing Core Deliverable)

The proposal requires comparing RL policies against DP-optimal. `portfolio_dp.py` exists in the project directory.

**Implementation:**
- Import or adapt the Phase 2 DP solution to produce a strategy function `dp_strategy(t, W, y, z) -> (cons_frac, weights)`
- Add it as a baseline alongside Static 60/40, Equal 1/3, and Glidepath in `run.py`
- This gives us the **optimality gap** measurement the proposal requires

---

## 6. Finer Consumption Grid

Current grid: [0%, 2%, 4%, 6%, 8%, 10%, 12%, 14%, 16%, 18%, 20%] — 11 levels, uniform 2% steps.

The critique notes RL agents consume ~14-16% per period (too aggressive), partly because the grid is too coarse at the low end where baselines operate (4%).

**Proposed grid:** [0%, 1%, 2%, 3%, 4%, 5%, 8%, 12%, 16%, 20%] — 10 levels, finer at low end.

**Impact:** Changes N_CONS and N_ACTIONS. All agents need retraining. Update `config.py` CONS_FRACS.

---

## 7. Add Certainty Equivalents and Per-Regime Evaluation (P2)

- **Certainty equivalent:** Convert E[Utility] to a dollar amount: CE = ((1-γ) * E[U])^(1/(1-γ)). Makes utility comparison interpretable.
- **Per-regime breakdown:** Run simulations starting from each regime (bear/normal/bull) separately. Show how each agent adapts.
- Add both to comparison table output and to report.md.

---

## Implementation Order

1. **3A** Double DQN (trivial, do immediately)
2. **3B/4A** Reward normalization for DQN + PPO
3. **1** Ensure all models in all plots + bump SARSA episodes
4. **2** Auto-generated report.md
5. **5** DP baseline
6. **3C, 3D, 4B, 4C** Additional DQN/PPO tuning
7. **6** Finer consumption grid (requires full retrain)
8. **7** Certainty equivalents and per-regime evaluation
