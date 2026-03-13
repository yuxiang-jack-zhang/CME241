"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Interface that all agents must implement."""

    # Set to True in agents that use store()/train_step() (DQN, PPO).
    # Agents that use update() (LinearQ, SARSA) leave this False.
    uses_replay = False

    # Whether to randomize initial states during training.
    # Neural network agents benefit from state diversity; linear agents
    # can diverge when the state distribution is too broad.
    randomize_start = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""

    @abstractmethod
    def select_action(self, state) -> int:
        """Choose an action given the current state feature vector."""

    def update(self, state, action, reward, next_state, done):
        """Online update (e.g. semi-gradient Q-learning). Optional."""

    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer. Optional for replay-based agents."""

    def train_step(self) -> float:
        """Batch training step (e.g. DQN). Returns loss. Optional."""
        return 0.0

    @abstractmethod
    def decay_epsilon(self, episode):
        """Decay exploration rate."""
