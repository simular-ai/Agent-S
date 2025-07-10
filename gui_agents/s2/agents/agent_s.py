import json
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.agents.worker import Worker
from gui_agents.s2.agents.manager import Manager
from gui_agents.s2.utils.common_utils import Node
from gui_agents.utils import download_kb_data
from gui_agents.s2.core.engine import (
    OpenAIEmbeddingEngine,
    GeminiEmbeddingEngine,
    AzureOpenAIEmbeddingEngine,
)

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyautogui",
        observation_type: str = "a11y_tree",
        search_engine: str = "perplexica",
    ):
        """Initialize UIAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
            action_space: Type of action space to use (pyautogui, aci)
            observation_type: Type of observations to use (a11y_tree, mixed)
            engine: Search engine to use (perplexica, LLM)
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.engine = search_engine

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

    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory with task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        pass

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory with subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        pass


class AgentS2(UIAgent):
    """Agent that uses hierarchical planning and directed acyclic graph modeling for UI automation"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyautogui",
        observation_type: str = "mixed",
        search_engine: Optional[str] = None,
        memory_root_path: str = os.getcwd(),
        use_default_kb: bool = False,
        memory_folder_name: str = "kb_s2",
        kb_release_tag: str = "v0.2.2",
        embedding_engine_type: str = "openai",
        embedding_engine_params: Dict = {},
    ):
        """Initialize AgentS2

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            action_space: Type of action space to use (pyautogui, other)
            observation_type: Type of observations to use (a11y_tree, screenshot, mixed)
            search_engine: Search engine to use (LLM, perplexica)
            use_default_kb: True to use the default OpenAI kb.
            memory_root_path: Path to memory directory. Defaults to current working directory.
            memory_folder_name: Name of memory folder. Defaults to "kb_s2".
            kb_release_tag: Release tag for knowledge base. Defaults to "v0.2.2".
            embedding_engine_type: Embedding engine to use for knowledge base. Defaults to "openai". Supports "openai" and "gemini".
            embedding_engine_params: Parameters for embedding engine. Defaults to {}.
        """
        super().__init__(
            engine_params,
            grounding_agent,
            platform,
            action_space,
            observation_type,
            search_engine,
        )

        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name
        self.kb_release_tag = kb_release_tag

        # Initialize agent's knowledge base on user's current working directory.
        self.local_kb_path = os.path.join(
            self.memory_root_path, self.memory_folder_name
        )

        if use_default_kb:
            if not os.path.exists(os.path.join(self.local_kb_path, self.platform)):
                print("Downloading Agent S2's default knowledge base...")
                download_kb_data(
                    version="s2",
                    release_tag=kb_release_tag,
                    download_dir=self.local_kb_path,
                    platform=self.platform,
                )
                print(
                    f"Successfully completed download of knowledge base for version s2, tag {self.kb_release_tag}, platform {self.platform}."
                )
            else:
                print(
                    f"Path local_kb_path {self.local_kb_path} already exists. Skipping download."
                )
                print(
                    f"If you'd like to re-download the initial knowledge base, please delete the existing knowledge base at {self.local_kb_path}."
                )
                print(
                    "Note, the knowledge is continually updated during inference. Deleting the knowledge base will wipe out all experience gained since the last knowledge base download."
                )

        if embedding_engine_type == "openai":
            self.embedding_engine = OpenAIEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "gemini":
            self.embedding_engine = GeminiEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "azure":
            self.embedding_engine = AzureOpenAIEmbeddingEngine(
                **embedding_engine_params
            )

        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        # Initialize core components
        self.planner = Manager(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            search_engine=self.engine,
            platform=self.platform,
        )
        self.executor = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            platform=self.platform,
        )

        # Reset state variables
        self.requires_replan: bool = True
        self.needs_next_subtask: bool = True
        self.step_count: int = 0
        self.turn_count: int = 0
        self.failure_subtask: Optional[Node] = None
        self.should_send_action: bool = False
        self.completed_tasks: List[Node] = []
        self.current_subtask: Optional[Node] = None
        self.subtasks: List[Node] = []
        self.search_query: str = ""
        self.subtask_status: str = "Start"

    def reset_executor_state(self) -> None:
        """Reset executor and step counter"""
        self.executor.reset()
        self.step_count = 0

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # Initialize the three info dictionaries
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions = []

        # If the DONE response by the executor is for a subtask, then the agent should continue with the next subtask without sending the action to the environment
        while not self.should_send_action:
            self.subtask_status = "In"
            # If replan is true, generate a new plan. True at start, after a failed plan, or after subtask completion
            if self.requires_replan:
                logger.info("(RE)PLANNING...")
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    observation=observation,
                    failed_subtask=self.failure_subtask,
                    completed_subtasks_list=self.completed_tasks,
                    remaining_subtasks_list=self.subtasks,
                )

                self.requires_replan = False
                if "search_query" in planner_info:
                    self.search_query = planner_info["search_query"]
                else:
                    self.search_query = ""

            # use the exectuor to complete the topmost subtask
            if self.needs_next_subtask:
                logger.info("GETTING NEXT SUBTASK...")

                # this can be empty if the DAG planner deems that all subtasks are completed
                if len(self.subtasks) <= 0:
                    self.requires_replan = True
                    self.needs_next_subtask = True
                    self.failure_subtask = None
                    self.completed_tasks.append(self.current_subtask)

                    # reset executor state
                    self.reset_executor_state()
                    self.should_send_action = True
                    self.subtask_status = "Done"
                    executor_info = {
                        "executor_plan": "agent.done()",
                        "plan_code": "agent.done()",
                        "reflection": "agent.done()",
                    }
                    actions = ["DONE"]
                    break

                self.current_subtask = self.subtasks.pop(0)
                logger.info(f"NEXT SUBTASK: {self.current_subtask}")
                self.needs_next_subtask = False
                self.subtask_status = "Start"

            # get the next action from the executor
            executor_info, actions = self.executor.generate_next_action(
                instruction=instruction,
                search_query=self.search_query,
                subtask=self.current_subtask.name,
                subtask_info=self.current_subtask.info,
                future_tasks=self.subtasks,
                done_task=self.completed_tasks,
                obs=observation,
            )

            self.step_count += 1

            # set the should_send_action flag to True if the executor returns an action
            self.should_send_action = True

            # replan on failure
            if "FAIL" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True

                # assign the failed subtask
                self.failure_subtask = self.current_subtask

                # reset the step count, executor, and evaluator
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False

            # replan on subtask completion
            elif "DONE" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True
                self.failure_subtask = None
                self.completed_tasks.append(self.current_subtask)

                # reset the step count, executor, and evaluator
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False
                self.subtask_status = "Done"

            self.turn_count += 1

        # reset the should_send_action flag for next iteration
        self.should_send_action = False

        # concatenate the three info dictionaries
        info = {
            **{
                k: v
                for d in [planner_info or {}, executor_info or {}, evaluator_info or {}]
                for k, v in d.items()
            }
        }
        info.update(
            {
                "subtask": self.current_subtask.name,
                "subtask_info": self.current_subtask.info,
                "subtask_status": self.subtask_status,
            }
        )

        return info, actions

    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory from task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        try:
            reflection_path = os.path.join(
                self.local_kb_path, self.platform, "narrative_memory.json"
            )
            try:
                reflections = json.load(open(reflection_path))
            except:
                reflections = {}

            if self.search_query not in reflections:
                reflection = self.planner.summarize_narrative(trajectory)
                reflections[self.search_query] = reflection

            with open(reflection_path, "w") as f:
                json.dump(reflections, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update narrative memory: {e}")

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory from subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        subtask = meta_data["subtask"]
        subtask_info = meta_data["subtask_info"]
        subtask_status = meta_data["subtask_status"]
        # Handle subtask trajectory
        if subtask_status == "Start" or subtask_status == "Done":
            # If it's a new subtask start, finalize the previous subtask trajectory if it exists
            if subtask_trajectory:
                subtask_trajectory += "\nSubtask Completed.\n"
                subtask_key = subtask_trajectory.split(
                    "\n----------------------\n\nPlan:\n"
                )[0]
                try:
                    subtask_path = os.path.join(
                        self.local_kb_path, self.platform, "episodic_memory.json"
                    )
                    kb = json.load(open(subtask_path))
                except:
                    kb = {}
                if subtask_key not in kb.keys():
                    subtask_summarization = self.planner.summarize_episode(
                        subtask_trajectory
                    )
                    kb[subtask_key] = subtask_summarization
                else:
                    subtask_summarization = kb[subtask_key]
                logger.info("subtask_key: %s", subtask_key)
                logger.info("subtask_summarization: %s", subtask_summarization)
                with open(subtask_path, "w") as fout:
                    json.dump(kb, fout, indent=2)
                # Reset for the next subtask
                subtask_trajectory = ""
            # Start a new subtask trajectory
            subtask_trajectory = (
                "Task:\n"
                + self.search_query
                + "\n\nSubtask: "
                + subtask
                + "\nSubtask Instruction: "
                + subtask_info
                + "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )
        elif subtask_status == "In":
            # Continue appending to the current subtask trajectory if it's still ongoing
            subtask_trajectory += (
                "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )

        return subtask_trajectory
