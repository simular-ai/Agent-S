import json
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.memory.memory import Memory
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
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.engine = search_engine
        self.memory = Memory("agent_memory.json")

    def reset(self) -> None:
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        pass

    def update_narrative_memory(self, trajectory: str) -> None:
        pass

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        pass


class AgentS2(UIAgent):
    def __init__(
            self,
            engine_params: Dict,
            grounding_agent: ACI,
            platform: str = platform.system().lower(),
            action_space: str = "pyautogui",
            observation_type: str = "mixed",
            search_engine: Optional[str] = None,
            memory_root_path: str = os.getcwd(),
            memory_folder_name: str = "kb_s2",
            kb_release_tag: str = "v0.2.2",
            embedding_engine_type: str = "openai",
            embedding_engine_params: Dict = {},
    ):
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

        print("Downloading knowledge base initial Agent-S knowledge...")
        self.local_kb_path = os.path.join(
            self.memory_root_path, self.memory_folder_name
        )

        if not os.path.exists(os.path.join(self.local_kb_path, self.platform)):
            download_kb_data(
                version="s2",
                release_tag=kb_release_tag,
                download_dir=self.local_kb_path,
                platform=self.platform,
            )
            print(f"Downloaded KB for version s2, tag {self.kb_release_tag}, platform {self.platform}.")
        else:
            print(f"KB path {self.local_kb_path} exists. Skipping download.")

        if embedding_engine_type == "openai":
            self.embedding_engine = OpenAIEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "gemini":
            self.embedding_engine = GeminiEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "azure":
            self.embedding_engine = AzureOpenAIEmbeddingEngine(**embedding_engine_params)

        self.reset()

    def reset(self) -> None:
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

        self.requires_replan = True
        self.needs_next_subtask = True
        self.step_count = 0
        self.turn_count = 0
        self.failure_subtask = None
        self.should_send_action = False
        self.completed_tasks = []
        self.current_subtask = None
        self.subtasks = []
        self.search_query = ""
        self.subtask_status = "Start"

    def reset_executor_state(self) -> None:
        self.executor.reset()
        self.step_count = 0

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions = []

        while not self.should_send_action:
            self.subtask_status = "In"
            if self.requires_replan:
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    observation=observation,
                    failed_subtask=self.failure_subtask,
                    completed_subtasks_list=self.completed_tasks,
                    remaining_subtasks_list=self.subtasks,
                )
                self.requires_replan = False
                self.search_query = planner_info.get("search_query", "")

            if self.needs_next_subtask:
                if len(self.subtasks) <= 0:
                    self.requires_replan = True
                    self.needs_next_subtask = True
                    self.failure_subtask = None
                    self.completed_tasks.append(self.current_subtask)
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
                self.needs_next_subtask = False
                self.subtask_status = "Start"

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
            self.should_send_action = True

            if "FAIL" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True
                self.failure_subtask = self.current_subtask
                self.reset_executor_state()
                if self.subtasks:
                    self.should_send_action = False

            elif "DONE" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True
                self.failure_subtask = None
                self.completed_tasks.append(self.current_subtask)
                self.reset_executor_state()
                if self.subtasks:
                    self.should_send_action = False
                self.subtask_status = "Done"

            self.turn_count += 1

        self.should_send_action = False
        info = {**planner_info, **executor_info, **evaluator_info}
        info.update({
            "subtask": self.current_subtask.name,
            "subtask_info": self.current_subtask.info,
            "subtask_status": self.subtask_status,
        })

        safe_observation = {k: v for k, v in observation.items() if k != "screenshot"}

        self.memory.store(instruction, {
            "observation": safe_observation,
            "actions": actions,
            "subtask": self.current_subtask.name,
            "subtask_info": self.current_subtask.info,
            "status": self.subtask_status
        })

        return info, actions

    def update_narrative_memory(self, trajectory: str) -> None:
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
        subtask = meta_data["subtask"]
        subtask_info = meta_data["subtask_info"]
        subtask_status = meta_data["subtask_status"]

        if subtask_status in ["Start", "Done"]:
            if subtask_trajectory:
                subtask_trajectory += "\nSubtask Completed.\n"
                subtask_key = subtask_trajectory.split("\n----------------------\n\nPlan:\n")[0]
                try:
                    subtask_path = os.path.join(
                        self.local_kb_path, self.platform, "episodic_memory.json"
                    )
                    kb = json.load(open(subtask_path))
                except:
                    kb = {}

                if subtask_key not in kb:
                    subtask_summarization = self.planner.summarize_episode(subtask_trajectory)
                    kb[subtask_key] = subtask_summarization
                else:
                    subtask_summarization = kb[subtask_key]

                logger.info("subtask_key: %s", subtask_key)
                logger.info("subtask_summarization: %s", subtask_summarization)
                with open(subtask_path, "w") as fout:
                    json.dump(kb, fout, indent=2)
                subtask_trajectory = ""

            subtask_trajectory = (
                    "Task:\n" + self.search_query + "\n\nSubtask: " + subtask
                    + "\nSubtask Instruction: " + subtask_info
                    + "\n----------------------\n\nPlan:\n" + meta_data["executor_plan"] + "\n"
            )

        elif subtask_status == "In":
            subtask_trajectory += (
                    "\n----------------------\n\nPlan:\n" + meta_data["executor_plan"] + "\n"
            )

        return subtask_trajectory
