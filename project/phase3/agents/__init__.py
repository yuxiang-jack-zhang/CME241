"""Agent registry. Add new agents here."""

from .linear_q import LinearQAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .sarsa import SarsaAgent

AGENT_REGISTRY = {
    "linear_q": LinearQAgent,
    "sarsa": SarsaAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
}
