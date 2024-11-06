from typing import Dict
from agent_s.mllm.MultimodalAgent import LMMAgent
from agent_s.core.Knowledge import KnowledgeBase

class BaseModule:
    def __init__(self, engine_params: Dict, platform: str):
        self.engine_params = engine_params
        self.platform = platform
        
    def _create_agent(self, system_prompt: str = None) -> LMMAgent:
        """Create a new LMMAgent instance"""
        agent = LMMAgent(self.engine_params)
        if system_prompt:
            agent.add_system_prompt(system_prompt)
        return agent
