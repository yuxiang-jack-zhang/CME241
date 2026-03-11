#!/usr/bin/env python
"""Main entry point — trains agents, runs simulations, generates plots."""

import os
import sys
import warnings
import random
import numpy as np
import torch

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

from .config import N_ACTIONS
from .env import PortfolioEnv
from .agents import AGENT_REGISTRY
from .train import train_agent
from .simulate import (
    simulate_strategy, make_strategy,
    static_60_40, static_equal, glidepath_strategy,
)
from .visualize import (
    plot_learning_curves, plot_policy_heatmaps, plot_regime_comparison,
    plot_terminal_wealth, plot_wealth_trajectories, plot_utility_distributions,
    plot_allocation_stacks, plot_consumption_paths,
    print_comparison_table, print_risk_metrics,
)


PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
N_SIM = 2000

# Agent training configs
AGENT_CONFIGS = {
    "linear_q": dict(
        state_dim=8, n_actions=N_ACTIONS,
        alpha=5e-5, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=2500,
    ),
    "dqn": dict(
        state_dim=8, n_actions=N_ACTIONS,
        lr=5e-4, buffer_size=50000, batch_size=64,
        target_update_freq=100,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=4000,
    ),
}

TRAIN_EPISODES = {
    "linear_q": 4000,
    "dqn": 6000,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Train agents ──
    trained_agents = {}
    train_results = {}

    for key, AgentClass in AGENT_REGISTRY.items():
        config = AGENT_CONFIGS.get(key, {})
        n_eps = TRAIN_EPISODES.get(key, 4000)
        agent = AgentClass(**config)
        env = PortfolioEnv()
        print(f"\nTraining {agent.name} agent...")
        returns, avg, losses = train_agent(agent, env, n_eps, agent_name=agent.name,
                                           log_every=500)
        trained_agents[key] = agent
        train_results[key] = (returns, avg, losses)

    # ── 2. Plot learning curves ──
    lin_r, lin_a, _ = train_results["linear_q"]
    dqn_r, dqn_a, dqn_l = train_results["dqn"]
    plot_learning_curves(lin_r, lin_a, dqn_r, dqn_a, dqn_l, plot_dir=PLOT_DIR)
    print(f"\nSaved learning curves to {PLOT_DIR}/")

    # ── 3. Policy visualizations ──
    agents_list = list(trained_agents.values())
    plot_policy_heatmaps(agents_list, plot_dir=PLOT_DIR)
    # Regime comparison for DQN
    if "dqn" in trained_agents:
        plot_regime_comparison(trained_agents["dqn"], plot_dir=PLOT_DIR)

    # ── 4. Monte Carlo simulations ──
    strategy_names = []
    strategy_fns = []

    # RL agents
    for key, agent in trained_agents.items():
        strategy_names.append(agent.name)
        strategy_fns.append(make_strategy(agent))

    # Baselines
    baselines = [
        ("Static 60/40", static_60_40),
        ("Equal 1/3",    static_equal),
        ("Glidepath",    glidepath_strategy),
    ]
    for bname, bfn in baselines:
        strategy_names.append(bname)
        strategy_fns.append(bfn)

    sim_results = {}
    for name, fn in zip(strategy_names, strategy_fns):
        print(f"Simulating {name}...")
        w, c, a, u = simulate_strategy(fn, N_SIM)
        sim_results[name] = {"wealth": w, "cons": c, "allocs": a, "utility": u}
    print("Done.\n")

    # ── 5. Comparison tables ──
    print_comparison_table(strategy_names, sim_results, N_SIM)
    print_risk_metrics(strategy_names, sim_results)

    # ── 6. Plots ──
    plot_terminal_wealth(strategy_names, sim_results, N_SIM, plot_dir=PLOT_DIR)
    plot_wealth_trajectories(strategy_names, sim_results, plot_dir=PLOT_DIR)
    plot_utility_distributions(strategy_names, sim_results, plot_dir=PLOT_DIR)
    plot_allocation_stacks(strategy_names, sim_results, plot_dir=PLOT_DIR)
    plot_consumption_paths(strategy_names, sim_results, plot_dir=PLOT_DIR)
    print(f"All plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
