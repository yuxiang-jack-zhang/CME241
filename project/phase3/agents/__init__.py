"""Agent registry. Add new agents here."""

from .linear_q import LinearQAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .sarsa import SarsaAgent

# Order matters: agents train sequentially sharing the global RNG,
# so new agents go at the end to preserve existing reproducibility.
AGENT_REGISTRY = {
    "linear_q": LinearQAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "sarsa": SarsaAgent,
}
