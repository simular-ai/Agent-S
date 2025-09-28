import logging
import platform
from typing import Dict, List, Tuple, Optional

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.agents.worker import Worker
from gui_agents.s2_5.agents.accessibility_agent import AccessibilityAgent, AccessibilityConfig

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
    ):
        """Initialize UIAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass


class AgentS2_5(UIAgent):
    """Agent that uses no hierarchy for less inference time with integrated accessibility testing"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        enable_accessibility: bool = False,
        accessibility_config: Optional[AccessibilityConfig] = None,
    ):
        """Initialize a minimalist AgentS2 without hierarchy

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
            enable_accessibility: Enable accessibility and 508 compliance testing
            accessibility_config: Configuration for accessibility testing
        """

        super().__init__(engine_params, grounding_agent, platform)
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.enable_accessibility = enable_accessibility
        
        # Initialize accessibility agent if enabled
        self.accessibility_agent = None
        if enable_accessibility:
            config = accessibility_config or AccessibilityConfig()
            self.accessibility_agent = AccessibilityAgent(config)
            logger.info("Accessibility testing enabled")
        
        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        self.executor = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
        )

    def start_accessibility_session(self, app_or_url: str) -> Optional[str]:
        """
        Start accessibility testing session
        
        Args:
            app_or_url: Application name or URL being tested
            
        Returns:
            Session ID if accessibility enabled, None otherwise
        """
        if not self.accessibility_agent:
            logger.warning("Accessibility testing not enabled")
            return None
        
        return self.accessibility_agent.start_accessibility_session(
            app_or_url, self.platform
        )

    def end_accessibility_session(self) -> Optional[Dict]:
        """
        End accessibility testing session and generate reports
        
        Returns:
            Session summary if accessibility enabled, None otherwise
        """
        if not self.accessibility_agent:
            return None
        
        session = self.accessibility_agent.end_accessibility_session()
        if session:
            return {
                "session_id": session.session_id,
                "violations_found": len(session.violations),
                "compliance_score": session.compliance_score,
                "keyboard_tests_run": len(session.keyboard_tests),
                "evidence_captured": len(session.evidence)
            }
        return None

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # Get standard agent prediction
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )

        # Perform accessibility analysis if enabled
        accessibility_info = {}
        if self.accessibility_agent:
            try:
                # Extract accessibility tree if available
                accessibility_tree = None
                if hasattr(self.grounding_agent, 'get_accessibility_tree'):
                    accessibility_tree = self.grounding_agent.get_accessibility_tree(observation)
                
                # Analyze observation for accessibility issues
                accessibility_analysis = self.accessibility_agent.analyze_observation(
                    observation, accessibility_tree
                )
                
                accessibility_info = {
                    "accessibility_violations": accessibility_analysis.get("violations", []),
                    "accessibility_score": accessibility_analysis.get("compliance_score", 0.0),
                    "accessibility_recommendations": accessibility_analysis.get("recommendations", [])
                }
                
                # Log accessibility issues if found
                violations_count = len(accessibility_analysis.get("violations", []))
                if violations_count > 0:
                    logger.warning(f"Found {violations_count} accessibility violations")
                    
            except Exception as e:
                logger.error(f"Error during accessibility analysis: {e}")
                accessibility_info["accessibility_error"] = str(e)

        # Combine all info dictionaries
        info = {
            **{k: v for d in [executor_info or {}] for k, v in d.items()},
            **accessibility_info
        }

        return info, actions

    def run_keyboard_accessibility_tests(self, observation: Dict) -> Optional[Dict]:
        """
        Run comprehensive keyboard accessibility tests
        
        Args:
            observation: Current observation
            
        Returns:
            Test results if accessibility enabled, None otherwise
        """
        if not self.accessibility_agent:
            logger.warning("Accessibility testing not enabled")
            return None
        
        # Extract accessibility tree if available
        accessibility_tree = None
        if hasattr(self.grounding_agent, 'get_accessibility_tree'):
            accessibility_tree = self.grounding_agent.get_accessibility_tree(observation)
        
        return self.accessibility_agent.run_keyboard_accessibility_tests(
            observation, accessibility_tree
        )

    def get_accessibility_status(self) -> Optional[Dict]:
        """
        Get current accessibility testing status
        
        Returns:
            Status information if accessibility enabled, None otherwise
        """
        if not self.accessibility_agent:
            return None
        
        return self.accessibility_agent.get_accessibility_status()

    def generate_accessibility_report(self, session_id: str = None) -> Optional[Dict[str, str]]:
        """
        Generate accessibility report
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            Dictionary with paths to generated reports if accessibility enabled, None otherwise
        """
        if not self.accessibility_agent:
            logger.warning("Accessibility testing not enabled")
            return None
        
        return self.accessibility_agent.generate_accessibility_report(session_id)
