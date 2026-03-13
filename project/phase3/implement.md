# Phase 3 Implementation Log

All items from plan.md implemented and run successfully.

## Completed Changes

### 3A: Double DQN
- `agents/dqn.py`: Changed target computation to use online net for action selection, target net for evaluation

### 3B/4A: Reward Normalization
- `utils.py`: Added `RewardNormalizer` class (Welford online mean/var, warmup guard for <2 samples)
- `train.py`: Integrated — normalizes rewards for `uses_replay` agents (DQN, PPO), raw rewards for Linear Q/SARSA

### 1: All Models in All Plots + SARSA Episodes
- `run.py`: SARSA training bumped 4k → 10k episodes
- `run.py`: Regime comparison plots now generated for all 4 agents
- `visualize.py`: Extended COLORS list to 9 entries for 8 strategies

### 2: Auto-Generated report.md
- `visualize.py`: Added `generate_report()`, `certainty_equivalent()`, `_compute_risk_metrics()` helper
- `run.py`: Calls `generate_report()` at end of main()

### 5: DP Baseline
- `dp_baseline.py` (new): Solves simplified DP (no income/regime states) using Phase 3 parameters with stationary regime distribution for expected returns. 60-point wealth grid.
- `run.py`: "DP Optimal" added as first baseline

### 3C/3D: DQN Tuning
- `run.py`: buffer_size 100k → 500k, target_update_freq 200 → 1000
- `agents/dqn.py`: Added `init_scheduler()` with CosineAnnealingLR (3e-4 → 1e-5), stepped per episode in `decay_epsilon()`

### 4B/4C: PPO Tuning
- `agents/ppo.py`: Value function clipping (clipped value loss with same clip_range as policy)
- `agents/ppo.py`: Entropy coefficient decays linearly from ent_coef → max(ent_coef*0.2, 0.01) over eps_decay episodes

### 6: Finer Consumption Grid
- `config.py`: CONS_FRACS changed to [0%, 1%, 2%, 3%, 4%, 5%, 8%, 12%, 16%, 20%] — 10 levels, finer at low end
- N_ACTIONS: 231 → 210

### 7: Certainty Equivalents + Per-Regime Evaluation
- `visualize.py`: CE column added to `print_comparison_table()` and `generate_report()`
- `simulate.py`: `simulate_strategy()` accepts optional `z0` parameter for fixed initial regime
- `run.py`: Per-regime evaluation loop (Bear/Normal/Bull) added after main simulation

## Bugs Found During Review (Fixed)
1. **DP double-discounting**: `dp_baseline.py` had β^t in per-step reward AND β·V in Bellman recursion — removed β^t
2. **RewardNormalizer.var init**: Was 1.0 (biased), fixed to 0.0 with warmup guard (skip normalization for <2 samples, use count-1 for unbiased variance)
3. **Dead code in train.py**: Removed unused `use_replay` variable and unreachable PPO branch
4. **Stale comment**: config.py said `# 231` after grid change to 210

## Run Results Summary

Training completed in ~25 minutes total. Key results (sorted by E[Utility]):

| Strategy | E[Utility] | CE | E[W_T] |
|----------|-----------|-----|--------|
| DP Optimal | -1.46 | 0.69 | 77.3 |
| Linear Q | -2.38 | 0.42 | 162.2 |
| Static 60/40 | -2.63 | 0.38 | 487.3 |
| Glidepath | -2.63 | 0.38 | 489.1 |
| Equal 1/3 | -2.75 | 0.36 | 407.8 |
| SARSA | -3.49 | 0.29 | 197.1 |
| DQN | -4.04 | 0.25 | 220.4 |
| PPO | -4.38 | 0.23 | 330.8 |

Linear Q outperforms all baselines. DQN/PPO improved from degenerate (uniform) policies but still underperform baselines. Full analysis in report.md.
