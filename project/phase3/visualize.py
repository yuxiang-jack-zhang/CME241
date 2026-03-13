"""All plotting and reporting functions."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    T, GAMMA, BETA, W0, N_INCOME, N_REGIME, REGIME_LABELS,
    N_ACTIONS, CONS_FRACS, ALLOC_ARRAY,
    INCOME_VALS, INCOME_TRANS, REGIME_TRANS,
)
from .env import PortfolioEnv


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
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

def _greedy_action(agent, state):
    """Get the greedy action index from any agent type."""
    if hasattr(agent, 'greedy_action'):
        return agent.greedy_action(state)
    elif hasattr(agent, 'q_net'):
        from .agents.dqn import device
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            q = agent.q_net(s)
            return int(q.argmax(dim=1).item())
    elif hasattr(agent, 'q_values'):
        q = agent.q_values(state)
        return int(np.argmax(q))
    else:
        raise ValueError(f"Cannot extract greedy action from {type(agent)}")


def extract_policy_grid(agent, wealth_grid, t_grid, y=1, z=1):
    """Query agent for greedy action at each (t, W) point."""
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

            action = _greedy_action(agent, state)
            cf, weights = env_tmp.decode_action(action)
            stock_alloc[ti, wi] = weights[0]
            cons_frac_grid[ti, wi] = cf

    return stock_alloc, cons_frac_grid


# ── Plot functions ──

def plot_learning_curves(train_results, plot_dir="plots"):
    """Plot learning curves for all trained agents.

    train_results: dict mapping agent key -> (returns, running_avg, losses)
    """
    _setup()
    tab_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    n_agents = len(train_results)
    has_losses = any(len(v[2]) > 0 for v in train_results.values())
    n_cols = 3 if has_losses else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    for i, (key, (returns, avg, losses)) in enumerate(train_results.items()):
        c = tab_colors[i % len(tab_colors)]
        axes[0].plot(returns, alpha=0.08, color=c)
        axes[0].plot(avg, linewidth=2, color=c, label=key)
        axes[1].plot(avg, linewidth=2, color=c, label=key)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Return")
    axes[0].set_title("Training Returns (100-ep running avg)")
    axes[0].legend()
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Running Average Return")
    axes[1].set_title("Running Average Return")
    axes[1].legend()

    if has_losses:
        window = 100
        for i, (key, (_, _, losses)) in enumerate(train_results.items()):
            if not losses:
                continue
            c = tab_colors[i % len(tab_colors)]
            axes[2].plot(losses, alpha=0.2, color=c)
            if len(losses) > window:
                smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window-1, len(losses)), smooth,
                             linewidth=2, color=c, label=key)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Training Loss")
        axes[2].legend()

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


def plot_consumption_paths(strategy_names, sim_results, n_rl_agents=3,
                           plot_dir="plots"):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, name in enumerate(strategy_names[:n_rl_agents]):
        c = sim_results[name]["cons"]
        mean_c = c.mean(axis=0)
        p10 = np.percentile(c, 10, axis=0)
        p90 = np.percentile(c, 90, axis=0)
        color = COLORS[i % len(COLORS)]
        ax.plot(range(T), mean_c, linewidth=2, color=color, label=name)
        ax.fill_between(range(T), p10, p90, alpha=0.15, color=color)

    for i, name in enumerate(strategy_names[n_rl_agents:], start=n_rl_agents):
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
    print(f"{'='*90}")
    print(f"  STRATEGY COMPARISON ({n_sim:,} simulated paths)")
    print(f"{'='*90}")
    print(f"  {'Strategy':<14} {'E[Utility]':>10} {'CE':>8} {'E[W_T]':>8} {'Med[W_T]':>9} "
          f"{'Std[W_T]':>9} {'P10[W_T]':>9} {'P90[W_T]':>9} {'P(ruin)':>8}")
    print(f"  {'-'*84}")

    for name in strategy_names:
        r = sim_results[name]
        tw = r["wealth"][:, -1]
        u = r["utility"]
        u_valid = u[u > -1e10]
        mean_u = u_valid.mean() if len(u_valid) > 0 else float('nan')
        ce = certainty_equivalent(mean_u)
        p_ruin = np.mean(tw < ruin_threshold)
        print(f"  {name:<14} {mean_u:>10.4f} {ce:>8.2f} {tw.mean():>8.1f} {np.median(tw):>9.1f} "
              f"{tw.std():>9.1f} {np.percentile(tw, 10):>9.1f} {np.percentile(tw, 90):>9.1f} "
              f"{p_ruin:>7.1%}")
    print()


def _compute_risk_metrics(strategy_names, sim_results):
    """Compute risk metrics and return as a list of dicts."""
    metrics = []
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
        metrics.append({"name": name, "avg_mdd": avg_mdd, "c_sharpe": c_sharpe, "min_tw": min_tw})
    return metrics


def print_risk_metrics(strategy_names, sim_results):
    metrics = _compute_risk_metrics(strategy_names, sim_results)
    print(f"\n{'='*60}")
    print(f"  RISK METRICS")
    print(f"{'='*60}")
    print(f"  {'Strategy':<14} {'MaxDrawdown':>12} {'Sharpe(C)':>10} {'MinW_T':>8}")
    print(f"  {'-'*48}")

    for m in metrics:
        print(f"  {m['name']:<14} {m['avg_mdd']:>11.1%} {m['c_sharpe']:>10.2f} {m['min_tw']:>8.1f}")
    print()


def certainty_equivalent(mean_utility, gamma=GAMMA):
    """Convert E[Utility] to certainty-equivalent wealth."""
    if gamma == 1.0:
        return np.exp(mean_utility)
    val = (1 - gamma) * mean_utility
    if val <= 0:
        return 0.0
    return val ** (1.0 / (1.0 - gamma))


def generate_report(strategy_names, sim_results, n_sim, plot_dir="plots",
                    report_path=None):
    """Generate a markdown report summarizing setup, models, and results."""
    if report_path is None:
        report_path = os.path.join(os.path.dirname(plot_dir), "report.md")

    ruin_threshold = 10.0
    lines = []
    lines.append("# Phase 3 — RL Portfolio Allocation Report\n")
    lines.append("*Auto-generated by `run.py`*\n")

    # Environment setup
    lines.append("## Environment Setup\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Horizon (T) | {T} |")
    lines.append(f"| CRRA risk aversion (γ) | {GAMMA} |")
    lines.append(f"| Discount factor (β) | {BETA} |")
    lines.append(f"| Initial wealth (W₀) | {W0} |")
    lines.append(f"| Action space size | {N_ACTIONS} ({len(CONS_FRACS)} consumption × {len(ALLOC_ARRAY)} allocation) |")
    lines.append(f"| State dimension | 8 |")
    lines.append(f"| Income states | {N_INCOME} (vals: {[float(x) for x in INCOME_VALS]}) |")
    lines.append(f"| Market regimes | {N_REGIME} ({', '.join(REGIME_LABELS)}) |")
    lines.append(f"| Consumption fractions | {[round(float(x), 3) for x in CONS_FRACS]} |")
    lines.append(f"| Simulation paths | {n_sim:,} |")
    lines.append("")

    # Strategy comparison table
    lines.append("## Strategy Comparison\n")
    lines.append(f"| Strategy | E[Utility] | CE | E[W_T] | Med[W_T] | Std[W_T] | P10 | P90 | P(ruin) |")
    lines.append(f"|----------|-----------|-----|--------|----------|----------|-----|-----|---------|")

    for name in strategy_names:
        r = sim_results[name]
        tw = r["wealth"][:, -1]
        u = r["utility"]
        u_valid = u[u > -1e10]
        mean_u = u_valid.mean() if len(u_valid) > 0 else float('nan')
        ce = certainty_equivalent(mean_u)
        p_ruin = np.mean(tw < ruin_threshold)
        lines.append(
            f"| {name} | {mean_u:.4f} | {ce:.2f} | {tw.mean():.1f} | "
            f"{np.median(tw):.1f} | {tw.std():.1f} | "
            f"{np.percentile(tw, 10):.1f} | {np.percentile(tw, 90):.1f} | "
            f"{p_ruin:.1%} |"
        )
    lines.append("")

    # Risk metrics table
    metrics = _compute_risk_metrics(strategy_names, sim_results)
    lines.append("## Risk Metrics\n")
    lines.append(f"| Strategy | MaxDrawdown | Sharpe(C) | Min W_T |")
    lines.append(f"|----------|------------|-----------|---------|")
    for m in metrics:
        sharpe_str = f"{m['c_sharpe']:.2f}" if m['c_sharpe'] != float('inf') else "∞"
        lines.append(f"| {m['name']} | {m['avg_mdd']:.1%} | {sharpe_str} | {m['min_tw']:.1f} |")
    lines.append("")

    # Plots
    lines.append("## Plots\n")
    plot_files = [
        ("Learning Curves", "learning_curves.png"),
        ("Terminal Wealth", "terminal_wealth.png"),
        ("Wealth Trajectories", "wealth_trajectories.png"),
        ("Utility Distributions", "utility_distributions.png"),
        ("Allocation Stacks", "allocation_stacks.png"),
        ("Consumption Paths", "consumption_paths.png"),
    ]
    for title, fname in plot_files:
        fpath = os.path.join("plots", fname)
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({fpath})\n")

    # Policy heatmaps (per agent)
    lines.append("### Policy Heatmaps\n")
    for name in strategy_names:
        fname = f"policy_{name.lower().replace(' ', '_')}.png"
        fpath = os.path.join("plots", fname)
        if os.path.exists(os.path.join(os.path.dirname(report_path), "plots", fname)):
            lines.append(f"#### {name}\n")
            lines.append(f"![{name} Policy]({fpath})\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")
