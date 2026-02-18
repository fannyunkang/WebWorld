# config.py

from dataclasses import dataclass

@dataclass
class AgentConfig:
    backend: str = "dlc"
    model: str = "Qwen3-8B"
    temperature: float = 0.5
    max_tokens: int = 2048

@dataclass
class WorldModelConfig:
    backend: str = "dlc"           
    model: str = "WebWorld-8B"
    temperature: float = 0.7
    max_tokens: int = 8196

@dataclass
class AppConfig:
    agent: AgentConfig = None
    world_model: WorldModelConfig = None
    max_steps: int = 300
    wm_context_window: int = 10
    agent_history_window: int = 10
    step_delay: float = 1.0

    def __post_init__(self):
        if self.agent is None:
            self.agent = AgentConfig()
        if self.world_model is None:
            self.world_model = WorldModelConfig()
