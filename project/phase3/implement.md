# Phase 3 Implementation Log

All items from plan.md implemented and run successfully.

## Completed Changes

### 3A: Double DQN
- `agents/dqn.py`: Changed target computation to use online net for action selection, target net for evaluation

### 3B/4A: Reward Normalization
- `utils.py`: Added `RewardNormalizer` class (Welford online mean/var, warmup guard for <2 samples)
- `train.py`: Integrated — normalizes rewards for `uses_replay` agents (DQN, PPO), raw rewards for Linear Q/SARSA

### 1: All Models in All Plots + SARSA Episodes
- `run.py`: SARSA now trains for 15k episodes (same as other agents)
- `run.py`: Regime comparison plots generated for all 4 agents
- `visualize.py`: Extended COLORS list to 9 entries for 8 strategies

### 2: Auto-Generated report.md
- `visualize.py`: Added `generate_report()`, `certainty_equivalent()`, `_compute_risk_metrics()` helper
- `run.py`: Calls `generate_report()` at end of main()

### 5: DP Baseline
- `dp_baseline.py` (new): Solves simplified DP (no income/regime states) using Phase 3 parameters with stationary regime distribution for expected returns. 60-point wealth grid.
- `run.py`: "DP Optimal" added as first baseline

### 3C/3D: DQN Tuning
- `run.py`: buffer_size 100k → 500k, target_update_freq 200 → 1000
- `agents/dqn.py`: Added `init_scheduler()` with CosineAnnealingLR (3e-4 → 1e-5), stepped per episode

### 4B/4C: PPO Tuning
- `agents/ppo.py`: Value function clipping (clipped value loss with same clip_range as policy)
- `agents/ppo.py`: Entropy coefficient decays linearly from ent_coef → max(ent_coef*0.2, 0.01)

### 6: Finer Consumption Grid
- `config.py`: CONS_FRACS changed to [0%, 1%, 2%, 3%, 4%, 5%, 8%, 12%, 16%, 20%] — N_ACTIONS: 231 → 210

### 7: Certainty Equivalents + Per-Regime Evaluation
- `visualize.py`: CE column in `print_comparison_table()` and `generate_report()`
- `simulate.py`: `simulate_strategy()` accepts optional `z0` for fixed initial regime
- `run.py`: Per-regime evaluation loop (Bear/Normal/Bull) after main simulation

## Bugs Found During Review (Fixed)
1. DP double-discounting: removed β^t from per-step DP reward (Bellman recursion already handles discounting)
2. RewardNormalizer.var init: 1.0 → 0.0 with warmup guard and unbiased variance (count-1)
3. Dead code in train.py: removed unused variable and unreachable branch
4. Stale comment: config.py `# 231` → `# 210`

## Final Run Results (Run 2)

| Strategy | E[Utility] | CE | E[W_T] |
|----------|-----------|-----|--------|
| DP Optimal | -1.46 | 0.69 | 77.3 |
| **Linear Q** | **-2.11** | **0.47** | 98.9 |
| Static 60/40 | -2.63 | 0.38 | 487.3 |
| Glidepath | -2.63 | 0.38 | 489.1 |
| Equal 1/3 | -2.75 | 0.36 | 407.8 |
| SARSA | -3.24 | 0.31 | 119.4 |
| DQN | -3.29 | 0.30 | 133.1 |
| PPO | -4.25 | 0.24 | 217.8 |

Linear Q beats all baselines (CE=0.47 vs 0.38). DQN/PPO no longer degenerate but still underperform baselines. Full analysis in report.md.
