"""All plotting and reporting functions."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import T, N_INCOME, N_REGIME, REGIME_LABELS
from .env import PortfolioEnv


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ALLOC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ASSET_NAMES = ["Stocks", "Bonds", "Cash"]


def _setup():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


def _savefig(fig, plot_dir, name):
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Policy extraction helper ──

def extract_policy_grid(agent, wealth_grid, t_grid, y=1, z=1):
    """Query agent for greedy action at each (t, W) point."""
    from .agents.dqn import device
    env_tmp = PortfolioEnv()
    stock_alloc = np.zeros((len(t_grid), len(wealth_grid)))
    cons_frac_grid = np.zeros((len(t_grid), len(wealth_grid)))

    for ti, t in enumerate(t_grid):
        for wi, W in enumerate(wealth_grid):
            env_tmp.t = t
            env_tmp.W = W
            env_tmp.y = y
            env_tmp.z = z
            env_tmp.done = False
            state = env_tmp.featurize()

            if hasattr(agent, 'q_net'):
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q = agent.q_net(s)
                    action = int(q.argmax(dim=1).item())
            else:
                q = agent.q_values(state)
                action = int(np.argmax(q))

            cf, weights = env_tmp.decode_action(action)
            stock_alloc[ti, wi] = weights[0]
            cons_frac_grid[ti, wi] = cf

    return stock_alloc, cons_frac_grid


# ── Plot functions ──

def plot_learning_curves(lin_returns, lin_avg, dqn_returns, dqn_avg, dqn_losses,
                         plot_dir="plots"):
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(lin_returns, alpha=0.1, color="tab:blue")
    axes[0].plot(lin_avg, linewidth=2, color="tab:blue", label="Linear Q")
    axes[0].plot(dqn_returns, alpha=0.1, color="tab:orange")
    axes[0].plot(dqn_avg, linewidth=2, color="tab:orange", label="DQN")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Return")
    axes[0].set_title("Training Returns (100-ep running avg)")
    axes[0].legend()

    axes[1].plot(lin_avg, linewidth=2, color="tab:blue", label="Linear Q")
    axes[1].plot(dqn_avg, linewidth=2, color="tab:orange", label="DQN")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Running Average Return")
    axes[1].set_title("Running Average Return")
    axes[1].legend()

    if dqn_losses:
        axes[2].plot(dqn_losses, alpha=0.3, color="tab:orange")
        window = 100
        if len(dqn_losses) > window:
            smooth = np.convolve(dqn_losses, np.ones(window)/window, mode='valid')
            axes[2].plot(range(window-1, len(dqn_losses)), smooth, linewidth=2, color="tab:red")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("DQN Training Loss")

    plt.tight_layout()
    _savefig(fig, plot_dir, "learning_curves.png")


def plot_policy_heatmaps(agents, plot_dir="plots"):
    """Plot stock allocation and consumption heatmaps for each agent."""
    _setup()
    wealth_vis = np.linspace(10, 400, 40)
    t_vis = np.arange(0, T, 2)

    for agent in agents:
        name = agent.name
        stocks, cons = extract_policy_grid(agent, wealth_vis, t_vis, y=1, z=1)

        # Stock allocation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        im1 = axes[0].imshow(stocks * 100, aspect="auto", origin="lower",
                              extent=[wealth_vis[0], wealth_vis[-1], t_vis[0], t_vis[-1]],
                              cmap="RdYlGn", vmin=0, vmax=100)
        axes[0].set_xlabel("Wealth")
        axes[0].set_ylabel("Time step t")
        axes[0].set_title(f"{name}: Stock Allocation % (Normal/Medium)")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(cons * 100, aspect="auto", origin="lower",
                              extent=[wealth_vis[0], wealth_vis[-1], t_vis[0], t_vis[-1]],
                              cmap="YlOrRd", vmin=0, vmax=20)
        axes[1].set_xlabel("Wealth")
        axes[1].set_ylabel("Time step t")
        axes[1].set_title(f"{name}: Consumption Rate % (Normal/Medium)")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        _savefig(fig, plot_dir, f"policy_{name.lower().replace(' ', '_')}.png")


def plot_regime_comparison(agent, plot_dir="plots"):
    """Compare stock allocation across regimes for a given agent."""
    _setup()
    wealth_vis = np.linspace(10, 400, 40)
    t_vis = np.arange(0, T, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for zi, (ax, rname) in enumerate(zip(axes, REGIME_LABELS)):
        stocks, _ = extract_policy_grid(agent, wealth_vis, t_vis, y=1, z=zi)
        im = ax.imshow(stocks * 100, aspect="auto", origin="lower",
                       extent=[wealth_vis[0], wealth_vis[-1], t_vis[0], t_vis[-1]],
                       cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xlabel("Wealth")
        ax.set_ylabel("Time step t")
        ax.set_title(f"{agent.name} Stock Alloc: {rname} Regime")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"{agent.name} Policy Across Market Regimes (Medium Income)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig(fig, plot_dir, f"regime_comparison_{agent.name.lower().replace(' ', '_')}.png")


def plot_terminal_wealth(strategy_names, sim_results, n_sim, plot_dir="plots"):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(strategy_names):
        tw = sim_results[name]["wealth"][:, -1]
        ax.hist(tw, bins=50, alpha=0.35, label=name, color=COLORS[i % len(COLORS)], density=True)
    ax.set_xlabel("Terminal Wealth")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal Wealth Distribution ({n_sim:,} paths)")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, plot_dir, "terminal_wealth.png")


def plot_wealth_trajectories(strategy_names, sim_results, plot_dir="plots"):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))
    ts = np.arange(T + 1)
    for i, name in enumerate(strategy_names):
        w = sim_results[name]["wealth"]
        med = np.median(w, axis=0)
        p10 = np.percentile(w, 10, axis=0)
        p90 = np.percentile(w, 90, axis=0)
        color = COLORS[i % len(COLORS)]
        ax.plot(ts, med, label=name, color=color, linewidth=2)
        ax.fill_between(ts, p10, p90, alpha=0.1, color=color)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Wealth")
    ax.set_title("Median Wealth Trajectory (shaded = 10th-90th percentile)")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, plot_dir, "wealth_trajectories.png")


def plot_utility_distributions(strategy_names, sim_results, plot_dir="plots"):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))
    all_utils = np.concatenate([sim_results[n]["utility"] for n in strategy_names])
    valid = all_utils[all_utils > -1e10]
    u_min = valid.min() if len(valid) > 0 else -10
    u_max = valid.max() if len(valid) > 0 else 0
    bins = np.linspace(u_min, u_max, 50)

    for i, name in enumerate(strategy_names):
        u = sim_results[name]["utility"]
        u_valid = u[u > -1e10]
        if len(u_valid) > 0:
            ax.hist(u_valid, bins=bins, alpha=0.35, label=name,
                    color=COLORS[i % len(COLORS)], density=True)
    ax.set_xlabel("Total Discounted CRRA Utility")
    ax.set_ylabel("Density")
    ax.set_title("Utility Distribution by Strategy")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, plot_dir, "utility_distributions.png")


def plot_allocation_stacks(strategy_names, sim_results, plot_dir="plots"):
    _setup()
    n = len(strategy_names)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes_flat = axes.flatten() if n > 3 else (axes if n > 1 else [axes])

    for idx, name in enumerate(strategy_names):
        ax = axes_flat[idx]
        mean_alloc = sim_results[name]["allocs"].mean(axis=0)
        ax.stackplot(range(T), mean_alloc.T, labels=ASSET_NAMES,
                     colors=ALLOC_COLORS, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Portfolio weight")
        ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    for idx in range(len(strategy_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Average Asset Allocation Over Time", fontsize=13)
    plt.tight_layout()
    _savefig(fig, plot_dir, "allocation_stacks.png")


def plot_consumption_paths(strategy_names, sim_results, plot_dir="plots"):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot first two (RL agents) with confidence bands
    for i, name in enumerate(strategy_names[:2]):
        c = sim_results[name]["cons"]
        mean_c = c.mean(axis=0)
        p10 = np.percentile(c, 10, axis=0)
        p90 = np.percentile(c, 90, axis=0)
        color = COLORS[i % len(COLORS)]
        ax.plot(range(T), mean_c, linewidth=2, color=color, label=name)
        ax.fill_between(range(T), p10, p90, alpha=0.15, color=color)

    # Add baselines as dashed
    for i, name in enumerate(strategy_names[2:], start=2):
        c = sim_results[name]["cons"]
        ax.plot(range(T), c.mean(axis=0), '--', linewidth=1.5,
                color=COLORS[i % len(COLORS)], label=name)

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Consumption")
    ax.set_title("Consumption Paths: RL Agents vs Baselines")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, plot_dir, "consumption_paths.png")


def print_comparison_table(strategy_names, sim_results, n_sim):
    ruin_threshold = 10.0
    print(f"{'='*80}")
    print(f"  STRATEGY COMPARISON ({n_sim:,} simulated paths)")
    print(f"{'='*80}")
    print(f"  {'Strategy':<14} {'E[Utility]':>10} {'E[W_T]':>8} {'Med[W_T]':>9} "
          f"{'Std[W_T]':>9} {'P10[W_T]':>9} {'P90[W_T]':>9} {'P(ruin)':>8}")
    print(f"  {'-'*76}")

    for name in strategy_names:
        r = sim_results[name]
        tw = r["wealth"][:, -1]
        u = r["utility"]
        u_valid = u[u > -1e10]
        mean_u = u_valid.mean() if len(u_valid) > 0 else float('nan')
        p_ruin = np.mean(tw < ruin_threshold)
        print(f"  {name:<14} {mean_u:>10.4f} {tw.mean():>8.1f} {np.median(tw):>9.1f} "
              f"{tw.std():>9.1f} {np.percentile(tw, 10):>9.1f} {np.percentile(tw, 90):>9.1f} "
              f"{p_ruin:>7.1%}")
    print()


def print_risk_metrics(strategy_names, sim_results):
    print(f"\n{'='*60}")
    print(f"  RISK METRICS")
    print(f"{'='*60}")
    print(f"  {'Strategy':<14} {'MaxDrawdown':>12} {'Sharpe(C)':>10} {'MinW_T':>8}")
    print(f"  {'-'*48}")

    for name in strategy_names:
        r = sim_results[name]
        w = r["wealth"]
        c = r["cons"]

        drawdowns = []
        for path in w:
            peak = np.maximum.accumulate(path)
            dd = (peak - path) / np.maximum(peak, 1e-10)
            drawdowns.append(dd.max())
        avg_mdd = np.mean(drawdowns)

        mean_c = c.mean(axis=0)
        if mean_c.std() > 1e-10:
            c_sharpe = mean_c.mean() / mean_c.std()
        else:
            c_sharpe = float('inf')

        min_tw = w[:, -1].min()

        print(f"  {name:<14} {avg_mdd:>11.1%} {c_sharpe:>10.2f} {min_tw:>8.1f}")
    print()
