"""Discrete-action PPO agent with actor-critic networks."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    """On-policy buffer for PPO. Stores one rollout then is cleared after update."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)

    def get_tensors(self):
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.LongTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
            torch.FloatTensor(np.array(self.values)).to(device),
            torch.FloatTensor(np.array(self.rewards)).to(device),
            torch.FloatTensor(np.array(self.dones)).to(device),
        )


def _ortho_init(module, gain=np.sqrt(2)):
    """Orthogonal initialization (standard for PPO)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    """Separate actor and critic networks for discrete action spaces.

    Using independent networks avoids the shared-trunk problem where the
    critic's loss dominates and prevents the actor from learning
    state-dependent features.
    """

    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.actor[:-1].apply(lambda m: _ortho_init(m, np.sqrt(2)))
        _ortho_init(self.actor[-1], 0.01)  # small init for policy head
        self.critic[:-1].apply(lambda m: _ortho_init(m, np.sqrt(2)))
        _ortho_init(self.critic[-1], 1.0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_dist(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization with clipped surrogate objective."""

    uses_replay = True
    randomize_start = True

    def __init__(self, state_dim, n_actions, lr=3e-4, hidden=256,
                 rollout_steps=512, ppo_epochs=4, minibatch_size=64,
                 clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, gae_lambda=0.95, gamma=1.0,
                 epsilon_start=0.3, epsilon_end=0.0, epsilon_decay_episodes=3000):
        self.n_actions = n_actions
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay_episodes

        self.net = ActorCritic(state_dim, n_actions, hidden).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self._last_log_prob = 0.0
        self._last_value = 0.0

    @property
    def name(self) -> str:
        return "PPO"

    @property
    def actor(self):
        """Expose network for strategy wrappers and visualization."""
        return self.net

    def greedy_action(self, state) -> int:
        """Deterministic greedy action (highest probability)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, _ = self.net(s)
            return int(logits.argmax(dim=1).item())

    def select_action(self, state) -> int:
        """Sample action from the policy; cache log_prob and value for store()."""
        if np.random.random() < self.epsilon:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                _, value = self.net(s)
                self._last_value = value.item()
            action = np.random.randint(self.n_actions)
            self._last_log_prob = -np.log(self.n_actions)
            return action

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, value = self.net.get_dist(s)
            action = dist.sample()
            self._last_log_prob = dist.log_prob(action).item()
            self._last_value = value.item()
            return int(action.item())

    def store(self, state, action, reward, next_state, done):
        """Store transition along with cached log_prob and value."""
        self.buffer.push(state, action, self._last_log_prob,
                         self._last_value, reward, done)

    def train_step(self) -> float:
        """Run PPO update when buffer is full. Returns mean policy loss."""
        if len(self.buffer) < self.rollout_steps:
            return 0.0

        states, actions, old_log_probs, old_values, rewards, dones = \
            self.buffer.get_tensors()

        # ── Compute GAE advantages ──
        with torch.no_grad():
            _, values_all = self.net(states)
            # Bootstrap value for the last state (0 if terminal)
            next_value = 0.0 if dones[-1] else values_all[-1].item()

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = old_values[t + 1].item()
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + old_values

        # Normalise advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO epochs with minibatches ──
        n = len(states)
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                dist, values = self.net.get_dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += policy_loss.item()
                n_updates += 1

        self.buffer.clear()
        return total_loss / max(n_updates, 1)

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(episode / self.eps_decay, 1.0)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
