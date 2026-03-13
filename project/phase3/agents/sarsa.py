"""SARSA agent with linear function approximation (on-policy TD control)."""

import numpy as np
from .base import BaseAgent


class SarsaAgent(BaseAgent):
    """Linear function approximation SARSA agent.

    Key difference from LinearQAgent (Q-learning): SARSA is on-policy —
    the TD target uses Q(s', a') where a' is the action actually selected
    by the epsilon-greedy policy, not max_a Q(s', a).

    Update rule (Eq. 12.5 from the textbook):
        Δw = α · (R + γ·Q(S',A';w) - Q(S,A;w)) · ∇_w Q(S,A;w)

    To ensure A' in the update is the SAME action actually taken next,
    update() picks A' and caches it; the next select_action() returns
    that cached action instead of sampling a fresh one.
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

        # Polynomial features: original + squared + pairwise of continuous features
        # State dim = 8 (2 continuous + 6 one-hot)
        # Features: 8 original + 2 squared + 1 cross + bias = 12
        self.feat_dim = state_dim + 3 + 1
        self.weights = np.zeros((n_actions, self.feat_dim))

        # Cache for the next action: ensures A' used in SARSA update
        # is the same action actually executed in the next step.
        self._cached_next_action = None

    @property
    def name(self) -> str:
        return "SARSA"

    def _expand_features(self, state):
        """Expand raw state features with polynomial terms."""
        t_feat, w_feat = state[0], state[1]
        poly = np.array([t_feat**2, w_feat**2, t_feat * w_feat])
        return np.concatenate([state, poly, [1.0]])

    def q_values(self, state):
        """Compute Q(s, a) for all actions."""
        phi = self._expand_features(state)
        return self.weights @ phi

    def _epsilon_greedy_action(self, state):
        """Pick action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_values(state)
        return int(np.argmax(q))

    def select_action(self, state) -> int:
        """Epsilon-greedy action selection.

        If update() already chose A' for this state (cached), return that
        to maintain the SARSA invariant. Otherwise sample fresh.
        """
        if self._cached_next_action is not None:
            action = self._cached_next_action
            self._cached_next_action = None
            return action
        return self._epsilon_greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        """On-policy SARSA update: Δw = α·(R + γ·Q(S',A') - Q(S,A))·∇Q(S,A).

        Picks A' from next_state using epsilon-greedy and caches it so that
        the next call to select_action(next_state) returns this same A'.
        """
        phi = self._expand_features(state)
        q_sa = self.weights[action] @ phi

        if done:
            target = reward
        else:
            # Choose A' on-policy and cache it for the next select_action call
            next_action = self._epsilon_greedy_action(next_state)
            self._cached_next_action = next_action
            q_next = self.weights[next_action] @ self._expand_features(next_state)
            target = reward + self.gamma_discount * q_next

        td_error = target - q_sa
        self.weights[action] += self.alpha * td_error * phi

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(episode / self.eps_decay, 1.0)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
