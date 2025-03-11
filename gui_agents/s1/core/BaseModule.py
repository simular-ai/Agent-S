from typing import Dict, Optional

from gui_agents.s1.mllm.MultimodalAgent import LMMAgent


class BaseModule:
    def __init__(self, engine_params: Dict, platform: str):
        self.engine_params = engine_params
        self.platform = platform

    def _create_agent(
        self, system_prompt: str = None, engine_params: Optional[Dict] = None
    ) -> LMMAgent:
        """Create a new LMMAgent instance"""
        agent = LMMAgent(engine_params or self.engine_params)
        if system_prompt:
            agent.add_system_prompt(system_prompt)
        return agent
