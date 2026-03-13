"""Agent-agnostic training loop."""

import time
import numpy as np
from .utils import RewardNormalizer


def train_agent(agent, env, n_episodes, agent_name="Agent", log_every=500):
    """Train an RL agent and return episode returns."""
    episode_returns = []
    running_avg = []
    losses = []
    t0 = time.time()

    # Reward normalization for neural network agents (DQN, PPO)
    uses_replay = getattr(agent, 'uses_replay', False)
    reward_normalizer = RewardNormalizer() if uses_replay else None
    randomize = getattr(agent, 'randomize_start', True)

    for ep in range(n_episodes):
        state = env.reset(randomize=randomize)
        ep_return = 0.0
        ep_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            ep_return += reward

            # Normalize reward for neural net agents
            if reward_normalizer is not None:
                reward_normalizer.update(reward)
                norm_reward = reward_normalizer.normalize(reward)
            else:
                norm_reward = reward

            if uses_replay:
                # Replay-based agent (DQN, PPO): store normalized reward
                agent.store(state, action, norm_reward, next_state, done)
                loss = agent.train_step()
                ep_loss += loss
            else:
                # Online agent (Linear Q, SARSA): immediate update with raw reward
                agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1
            if done:
                break

        episode_returns.append(ep_return)
        avg = np.mean(episode_returns[-100:])
        running_avg.append(avg)
        if uses_replay:
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
