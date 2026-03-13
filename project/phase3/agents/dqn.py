"""DQN agent with target network and experience replay."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .base import BaseAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size circular replay buffer."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def _ortho_init(module, gain=np.sqrt(2)):
    """Orthogonal initialization."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class QNetwork(nn.Module):
    """Dueling Q-network: separate value and advantage streams."""
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, n_actions)
        self.feature.apply(lambda m: _ortho_init(m, np.sqrt(2)))
        _ortho_init(self.value_stream, 1.0)
        _ortho_init(self.advantage_stream, 0.01)

    def forward(self, x):
        h = self.feature(x)
        v = self.value_stream(h)
        a = self.advantage_stream(h)
        return v + a - a.mean(dim=1, keepdim=True)


class DQNAgent(BaseAgent):
    """Dueling DQN agent with target network and experience replay."""

    uses_replay = True

    def __init__(self, state_dim, n_actions, lr=1e-3, hidden=256, buffer_size=50000,
                 batch_size=64, target_update_freq=50, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=3000):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay_episodes

        self.q_net = QNetwork(state_dim, n_actions, hidden).to(device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.update_count = 0

    @property
    def name(self) -> str:
        return "DQN"

    def select_action(self, state) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Sample a batch and do one gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(states).to(device)
        a = torch.LongTensor(actions).to(device)
        r = torch.FloatTensor(rewards).to(device)
        ns = torch.FloatTensor(next_states).to(device)
        d = torch.FloatTensor(dones).to(device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1)[0]
            target = r + self.gamma * (1 - d) * q_next

        loss = self.loss_fn(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(episode / self.eps_decay, 1.0)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
