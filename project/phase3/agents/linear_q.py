"""Linear function approximation Q-learning agent."""

import numpy as np
from .base import BaseAgent


class LinearQAgent(BaseAgent):
    """Linear function approximation Q-learning with rich feature expansion.

    Features include polynomial terms of continuous features (time, wealth)
    AND interactions between continuous features and one-hot categorical
    features (income, regime), so the agent can learn different policies
    for different regimes/income states.
    """

    def __init__(self, state_dim, n_actions, alpha=1e-4, gamma_discount=1.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=2000):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma_discount = gamma_discount
        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay_episodes

        # state = [t_feat, w_feat, y0, y1, y2, z0, z1, z2]  (dim=8)
        # Expanded features:
        #   8 original + 3 polynomial (t^2, w^2, t*w)
        #   + 6 interactions: t*y0..y2, t*z0..z2
        #   + 6 interactions: w*y0..y2, w*z0..z2
        #   + 1 bias = 24
        self.feat_dim = 24
        self.weights = np.zeros((n_actions, self.feat_dim))

    @property
    def name(self) -> str:
        return "Linear Q"

    def _expand_features(self, state):
        """Expand raw state features with polynomial and interaction terms."""
        t, w = state[0], state[1]
        cat = state[2:]  # 6 one-hot values (3 income + 3 regime)
        poly = np.array([t**2, w**2, t * w])
        t_cross = t * cat   # 6 terms: time interacted with each categorical
        w_cross = w * cat   # 6 terms: wealth interacted with each categorical
        return np.concatenate([state, poly, t_cross, w_cross, [1.0]])

    def q_values(self, state):
        """Compute Q(s, a) for all actions."""
        phi = self._expand_features(state)
        return self.weights @ phi

    def select_action(self, state) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_values(state)
        return int(np.argmax(q))

    def update(self, state, action, reward, next_state, done):
        """Semi-gradient Q-learning update."""
        phi = self._expand_features(state)
        q_sa = self.weights[action] @ phi

        if done:
            target = reward
        else:
            q_next = self.q_values(next_state)
            target = reward + self.gamma_discount * np.max(q_next)

        td_error = target - q_sa
        self.weights[action] += self.alpha * td_error * phi

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(episode / self.eps_decay, 1.0)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
