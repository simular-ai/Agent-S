"""Cognition layer for Agent-S3 (TIER 4 + TIER 5 Critic)."""

from gui_agents.s3.cognition.critic_agent import CriticAgent, ReviewResult, SearchResult
from gui_agents.s3.cognition.self_healing import HealingResult, SelfHealingEngine

__all__ = [
    "CriticAgent",
    "HealingResult",
    "ReviewResult",
    "SearchResult",
    "SelfHealingEngine",
]