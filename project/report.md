# Lifecycle Dynamic Asset Allocation: DP and RL Approaches

## 1. Introduction

This project addresses the lifecycle portfolio allocation problem: an investor must decide how much to consume and how to allocate wealth across stocks, bonds, and cash at each period over a finite horizon, subject to stochastic income and market regimes. The objective is to maximize total discounted CRRA utility of consumption plus a terminal bequest.

The project proceeds in two phases. Phase 2 solves a simplified version exactly via Dynamic Programming (DP) on discrete grids, establishing a baseline and demonstrating why DP becomes infeasible at realistic scale. Phase 3 tackles the full-scale problem with four Reinforcement Learning algorithms — Linear Q-learning, SARSA, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO) — and documents the iterative process of debugging and improving these agents to learn meaningful, state-dependent policies.

---

## 2. Phase 2: Dynamic Programming

### 2.1 Problem Setup

| Parameter | Value |
|-----------|-------|
| Horizon | T = 25 periods |
| Risk aversion | γ = 2.0 (CRRA) |
| Discount factor | β = 0.96 |
| Wealth grid | 121 points, linearly spaced |
| Income states | 2 (Low = 5, High = 20), 2×2 Markov transition |
| Market regimes | 2 (Bear, Bull), 2×2 Markov transition |
| Consumption fractions | 11 choices: 0%, 2%, 4%, ..., 20% |
| Portfolio allocations | 15 three-asset combinations (25% increments) |
| Actions per state | 165 (11 × 15) |

The utility function is CRRA: u(c) = c^(1−γ) / (1−γ) for γ ≠ 1. With γ = 2, this gives u(c) = −1/c.

Wealth transitions follow:

W' = (W − c) · (w · R) + income(y')

where c is consumption, w is the portfolio weight vector, and R is the vector of gross asset returns sampled from regime-dependent scenarios.

### 2.2 Solver

Finite-horizon backward induction computes the value function V_t(W, y, z) for each state at each time step, starting from the terminal condition V_T(W) = u(W). The solver is fully vectorized in NumPy and runs in under 1 second.

### 2.3 Results

Simulating 2,000 paths from initial wealth W₀ = 100:

| Strategy | E[Utility] | E[W_T] |
|----------|-----------|--------|
| **DP-Optimal** | **−0.95** | 88.5 |
| Static 60/40 | −2.23 | 351.2 |
| Glidepath | −2.22 | 354.1 |
| Equal 1/3 | −2.35 | — |

The DP-optimal policy achieves the highest utility despite lower terminal wealth — it consumes more along the path, which is exactly what the CRRA objective rewards. This validated the DP implementation and established the conceptual framework for Phase 3.

### 2.4 Dimensionality Analysis

Scaling to a realistic problem (T = 40, 500-point wealth grid, 3 income states, 3 regimes, 10% allocation increments) yields ~238 million DP cells — a 119× blowup from Phase 2's ~2 million. This makes exact DP infeasible and motivates the use of RL with function approximation in Phase 3.

---

## 3. Phase 3: Reinforcement Learning

### 3.1 Problem Setup (Expanded)

| Parameter | Value |
|-----------|-------|
| Horizon | T = 40 periods |
| Risk aversion | γ = 2.0 |
| Discount factor | β = 0.96 |
| Initial wealth | W₀ = 100 |
| Income states | 3 (Low = 3, Medium = 10, High = 25) |
| Market regimes | 3 (Bear, Normal, Bull) |
| Consumption fractions | 11 choices: 0%, 2%, ..., 20% |
| Portfolio allocations | 21 three-asset combinations (20% increments) |
| **Total actions** | **231** (11 × 21) |

### 3.2 Environment

The `PortfolioEnv` class provides a Gym-like interface with:

- **State representation** (8-dimensional feature vector):
  - Normalized time: t/T
  - Normalized log-wealth: log(W) / log(W_MAX)
  - 3-dim one-hot encoding for income state
  - 3-dim one-hot encoding for market regime

- **Reward**: β^t · u(c_t) at each step, plus β^T · u(W_T) at the terminal step. The β^t factor preserves the economic time value of money.

- **Transitions**: Stochastic income and regime via Markov chains, stochastic asset returns via regime-dependent scenarios.

### 3.3 The Four RL Agents

#### 3.3.1 Linear Q-Learning (Off-Policy)

**Architecture:** Linear function approximation Q(s, a) = w_a · φ(s), where φ(s) is a 24-dimensional feature vector:
- 8 raw state features
- 3 polynomial terms (t², w², t·w)
- 6 time-categorical interactions (t × each one-hot)
- 6 wealth-categorical interactions (w × each one-hot)
- 1 bias term

The interaction terms are critical — they allow the linear model to learn different policies for different income states and market regimes.

**Update rule (semi-gradient Q-learning):**
w_a ← w_a + α · (r + γ · max_{a'} Q(s', a') − Q(s, a)) · φ(s)

**Hyperparameters:** α = 10⁻⁵, γ = 1.0, ε: 1.0 → 0.05 over 8,000 episodes, 15,000 total training episodes.

#### 3.3.2 SARSA (On-Policy)

**Architecture:** Identical to Linear Q (24-dim features, linear Q-function).

**Key difference:** SARSA uses the on-policy TD target Q(s', a') where a' is the action actually taken by the ε-greedy policy, rather than max_a Q(s', a). A cached-action mechanism ensures that the a' used in the update is the same action executed at the next step.

**Update rule:**
w_a ← w_a + α · clip(r + γ · Q(s', a') − Q(s, a), −10, 10) · φ(s)

TD-error clipping at ±10 prevents gradient explosions from the large CRRA penalties.

**Hyperparameters:** Same as Linear Q.

#### 3.3.3 Dueling DQN (Off-Policy)

**Architecture:** Dueling Q-network with separate value and advantage streams:
- Shared feature extractor: 8 → 256 → Tanh → 256 → Tanh
- Value stream: 256 → 1
- Advantage stream: 256 → 231
- Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

Orthogonal initialization with gain √2 for hidden layers, gain 0.01 for the advantage head, and gain 1.0 for the value head.

**Training:** Experience replay buffer (100K capacity), target network (updated every 200 steps), Huber loss, gradient clipping at norm 10.

**Hyperparameters:** lr = 3×10⁻⁴, batch size = 128, γ = 1.0, ε: 1.0 → 0.05 over 8,000 episodes, 15,000 total episodes.

#### 3.3.4 PPO (On-Policy, Actor-Critic)

**Architecture:** Separate actor and critic networks (no shared trunk):
- Actor: 8 → 256 → Tanh → 256 → Tanh → 231 (logits)
- Critic: 8 → 256 → Tanh → 256 → Tanh → 1 (value)

Orthogonal initialization with gain 0.01 for the policy head to start with a near-uniform distribution.

**Training:** On-policy rollout buffer (2,048 steps), Generalized Advantage Estimation (λ = 0.95), clipped surrogate objective (ε = 0.2), entropy bonus coefficient = 0.05, 10 PPO epochs per rollout, minibatch size 128.

**Hyperparameters:** lr = 3×10⁻⁴, γ = 1.0, ε-greedy: 0.3 → 0.0 over 5,000 episodes, 15,000 total episodes.

### 3.4 Training Infrastructure

The agent-agnostic training loop dispatches based on an explicit `uses_replay` flag:
- **Batch agents** (DQN, PPO): call `store()` then `train_step()` each step
- **Online agents** (Linear Q, SARSA): call `update()` each step

All agents train with **randomized initial conditions** — each episode starts with random wealth (log-uniform between 10 and 480), random income state, and random market regime. This is essential for learning state-dependent policies, as fixed initial conditions (always W₀ = 100, medium income, normal regime) would only expose the agent to a narrow slice of the state space.

---

## 4. Iterative Improvements

Getting RL agents to learn meaningful lifecycle policies required solving several non-trivial bugs and design issues, addressed iteratively:

### 4.1 Zero-Consumption Exploit (Reward Bug)

**Problem:** With CRRA γ = 2, u(c) = −1/c is always negative for c > 0. The original environment returned reward = 0 for zero consumption, making it strictly better than any positive consumption. All agents rationally learned to never consume.

**Fix:** Zero consumption now receives the full CRRA penalty (u(0) = −10⁴), matching Phase 2's DP treatment where u(0) = −10¹⁸. The penalty was moderated from −10¹² to −10⁴ for neural network training stability.

### 4.2 Training Loop Dispatch Bug

**Problem:** The training loop used `hasattr(agent, 'store')` to distinguish batch from online agents. Since `BaseAgent` defines a no-op `store()`, this was always `True` — Linear Q's `update()` method was never called. Linear Q learned nothing for the entirety of early experiments.

**Fix:** Replaced with an explicit class-level `uses_replay` flag. DQN and PPO set `uses_replay = True`; Linear Q and SARSA inherit the default `False`.

### 4.3 Simulation Utility Bug

**Problem:** The Monte Carlo simulation in `simulate.py` had the same zero-consumption bug — it skipped adding utility for periods with zero consumption, making non-consuming agents appear to have near-zero (good) utility.

**Fix:** Removed the `if c > 1e-10` guard so `crra_scalar(c)` is always called, applying the −10⁴ penalty for zero consumption.

### 4.4 Architecture Improvements

**Shared vs. Separate Actor-Critic (PPO):** The original PPO used a shared-trunk network where the critic's value loss dominated, preventing the actor from learning state-dependent features. Splitting into independent actor and critic networks with separate parameters fixed this.

**Dueling Architecture (DQN):** Upgraded from a plain Q-network to a dueling architecture that separately estimates state value and action advantages. This helps the network learn which states are valuable independently of the action taken.

**Rich Feature Expansion (Linear Q, SARSA):** Expanding the 8-dimensional state to 24 dimensions with polynomial terms and cross-interactions (time×regime, wealth×income) allows the linear model to represent state-dependent policies despite having no hidden layers.

### 4.5 Training Improvements

**Randomized Initial Conditions:** Training with random starting wealth, income, and regime ensures the agent experiences diverse states, preventing overfitting to a single trajectory. This was the single most impactful change for learning state-dependent policies.

**Larger Networks:** Hidden layer size increased from 128 to 256 for both DQN and PPO.

**Higher Entropy Bonus (PPO):** Increased from 0.01 to 0.05 to encourage exploration across the 231-action space.

**Larger Rollout Buffer (PPO):** Increased from 512 to 2,048 steps per rollout for more diverse training data.

**More Training Episodes:** All agents train for 15,000 episodes (up from 4,000–6,000).

---

## 5. Results

### 5.1 Strategy Comparison (2,000 Simulated Paths)

| Strategy | E[Utility] | E[W_T] | Med[W_T] | Std[W_T] | P10[W_T] | P90[W_T] |
|----------|-----------|--------|----------|----------|----------|----------|
| **DQN** | **−1.54** | 115.3 | 107.2 | 51.8 | 57.9 | 185.5 |
| **Linear Q** | **−1.59** | 96.0 | 86.6 | 33.8 | 59.8 | 144.2 |
| **SARSA** | **−1.88** | 99.1 | 93.3 | 35.3 | 58.1 | 149.9 |
| **PPO** | **−1.95** | 91.0 | 81.2 | 43.2 | 44.8 | 149.8 |
| Static 60/40 | −2.63 | 487.3 | 470.1 | 166.1 | 281.0 | 747.3 |
| Glidepath | −2.63 | 489.1 | 474.1 | 140.0 | 319.7 | 696.5 |
| Equal 1/3 | −2.75 | 407.8 | 396.1 | 112.4 | 270.5 | 564.4 |

All four RL agents outperform all three baselines on expected CRRA utility. The RL agents achieve lower terminal wealth because they consume more along the path — precisely the behavior the CRRA objective rewards.

### 5.2 Risk Metrics

| Strategy | Max Drawdown | Sharpe(C) | Min W_T |
|----------|-------------|-----------|---------|
| DQN | 60.4% | 26.53 | 18.3 |
| Linear Q | 54.0% | 18.30 | 33.2 |
| SARSA | 57.1% | 10.25 | 24.9 |
| PPO | 66.1% | 12.58 | 20.5 |
| Static 60/40 | 23.8% | 2.65 | 98.4 |

The RL agents have higher consumption Sharpe ratios (smoother consumption relative to mean) but accept higher maximum drawdowns and lower minimum terminal wealth. This reflects the fundamental tradeoff: aggressive consumption smoothing reduces the wealth buffer.

### 5.3 Learned Policies

**Linear Q** learned the richest state-dependent policy:
- Stock allocation decreases from ~80% early to ~40% late — a learned glidepath.
- Consumption rate increases with time (2–6% early, 15–20% late) and decreases with wealth.
- The allocation stacks show a diversified, time-varying portfolio (stocks, bonds, cash all present).

**SARSA** learned a similar but distinct pattern:
- More aggressive early stock allocation at low wealth, with a clear time-wealth gradient.
- Consumption shows a staircase pattern varying with both time and wealth.
- The on-policy nature produces a more conservative policy than off-policy Q-learning.

**DQN** converged to an aggressive constant policy: ~100% stocks and ~20% consumption. This policy achieves the best expected utility by exploiting the equity premium over 40 periods, though it doesn't differentiate across states.

**PPO** learned a regime-dependent policy: bear, normal, and bull market panels show distinctly different stock allocations. It uses lower stock allocation early (~10%) and higher later (~20–30%), with consumption increasing over time.

---

## 6. Discussion

### 6.1 Linear Models vs. Neural Networks

A surprising finding is that the linear agents (Linear Q and SARSA) produced more visually state-dependent policies than the neural network agents (DQN and PPO). This is partly because the 24-dimensional feature expansion explicitly encodes the interaction terms (time×regime, wealth×income) that the policy should respond to. The neural networks must learn these interactions implicitly from data, which requires more training.

DQN achieves the best utility despite a constant-looking policy because its action (100% stocks, 20% consumption) happens to be near-optimal across most states for a CRRA-2 investor with a 40-year horizon. The equity premium dominates at this time scale.

### 6.2 On-Policy vs. Off-Policy

Linear Q (off-policy) slightly outperforms SARSA (on-policy) at −1.59 vs −1.88. Off-policy Q-learning can exploit experience more aggressively by always bootstrapping from the greedy action, while SARSA's on-policy updates are more conservative — the TD target reflects the actual exploratory behavior. SARSA's TD-error clipping at ±10 also limits the speed of value propagation, which may contribute to the gap.

### 6.3 The Role of β^t in Rewards

Including β^t in the environment's reward (rather than letting the RL agent's discount factor handle it) preserves the economic interpretation: early consumption is worth more than late consumption. The agents use γ = 1.0 internally so that the total return exactly equals the discounted CRRA objective. This design choice is deliberate — it keeps the RL objective aligned with the economic problem.

### 6.4 Limitations

- **Action space:** The flat 231-action space is inefficient. Decomposing into separate consumption and allocation heads (11 + 21 = 32 effective actions) would dramatically improve exploration.
- **Training stability:** PPO's on-policy nature combined with randomized initial conditions causes high variance in episode returns, leading to oscillating training curves.
- **Continuous actions:** The discrete allocation grid (20% increments) is coarse. Continuous-action methods (e.g., continuous PPO or SAC) could produce smoother, finer-grained policies.

---

## 7. Conclusion

This project demonstrated the full pipeline from exact Dynamic Programming to scalable Reinforcement Learning for lifecycle portfolio allocation. Phase 2 established that DP produces optimal policies on small grids but becomes infeasible at realistic scale (119× state-space blowup). Phase 3 showed that four different RL algorithms — spanning linear and deep, on-policy and off-policy methods — can all learn policies that outperform standard financial baselines (Static 60/40, Glidepath, Equal Weight) on the CRRA utility objective.

The iterative debugging process was itself instructive: a single misplaced `reward = 0.0` for zero consumption made all agents learn degenerate policies, and a broken dispatch mechanism silently prevented an entire agent class from learning. These issues are representative of the practical challenges in applying RL to financial problems, where reward shaping, numerical stability, and correct software engineering are as important as algorithm choice.

The best-performing agent (Dueling DQN, E[Utility] = −1.54) learned an aggressive equity-heavy strategy that exploits the long horizon, while the most interpretable agents (Linear Q and SARSA) learned rich state-dependent glidepath policies that vary meaningfully with time, wealth, income, and market regime. Together, they validate that RL is a viable and powerful approach to lifecycle asset allocation at scales where exact DP is no longer feasible.
