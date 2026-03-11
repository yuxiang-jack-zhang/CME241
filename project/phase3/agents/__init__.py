"""Agent registry. Add new agents here."""

from .linear_q import LinearQAgent
from .dqn import DQNAgent

AGENT_REGISTRY = {
    "linear_q": LinearQAgent,
    "dqn": DQNAgent,
}
