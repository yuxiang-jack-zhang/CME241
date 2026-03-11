"""Agent-agnostic training loop."""

import time
import numpy as np


def train_agent(agent, env, n_episodes, agent_name="Agent", log_every=500):
    """Train an RL agent and return episode returns."""
    episode_returns = []
    running_avg = []
    losses = []
    t0 = time.time()

    for ep in range(n_episodes):
        state = env.reset()
        ep_return = 0.0
        ep_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            ep_return += reward

            if hasattr(agent, 'store') and callable(getattr(agent, 'store')):
                # Replay-based agent (e.g. DQN): store transition and do batch update
                agent.store(state, action, reward, next_state, done)
                loss = agent.train_step()
                ep_loss += loss
            else:
                # Online agent (e.g. Linear Q): immediate update
                agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1
            if done:
                break

        episode_returns.append(ep_return)
        avg = np.mean(episode_returns[-100:])
        running_avg.append(avg)
        if hasattr(agent, 'store') and callable(getattr(agent, 'store')):
            losses.append(ep_loss / max(steps, 1))
        agent.decay_epsilon(ep)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{agent_name}] Ep {ep+1:>5d}/{n_episodes}  "
                  f"avg_return={avg:.2f}  eps={agent.epsilon:.3f}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  [{agent_name}] Training done in {elapsed:.1f}s  "
          f"final_avg={running_avg[-1]:.2f}")
    return episode_returns, running_avg, losses
