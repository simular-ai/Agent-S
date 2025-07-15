import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import platform

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.core.module import BaseModule
from gui_agents.s2.core.knowledge import KnowledgeBase
from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.core.engine import OpenAIEmbeddingEngine
from gui_agents.s2.utils.common_utils import (
    Dag,
    Node,
    calculate_tokens,
    call_llm_safe,
    parse_dag,
)

logger = logging.getLogger("desktopenv.agent")

NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


class Manager(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        local_kb_path: str,
        embedding_engine,
        search_engine: Optional[str] = None,
        multi_round: bool = False,
        platform: str = platform.system().lower(),
    ):
        # TODO: move the prompt to Procedural Memory
        super().__init__(engine_params, platform)

        # Initialize the ACI
        self.grounding_agent = grounding_agent

        # Initialize the planner
        sys_prompt = PROCEDURAL_MEMORY.COMBINED_MANAGER_PROMPT

        self.generator_agent = self._create_agent(sys_prompt)

        # Initialize the remaining modules
        self.dag_translator_agent = self._create_agent(
            PROCEDURAL_MEMORY.DAG_TRANSLATOR_PROMPT
        )
        self.narrative_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.TASK_SUMMARIZATION_PROMPT
        )
        self.episode_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )

        self.local_kb_path = local_kb_path

        self.embedding_engine = embedding_engine
        self.knowledge_base = KnowledgeBase(
            embedding_engine=self.embedding_engine,
            local_kb_path=self.local_kb_path,
            platform=platform,
            engine_params=engine_params,
        )

        self.planner_history = []

        self.turn_count = 0
        self.search_engine = search_engine
        self.multi_round = multi_round

    def summarize_episode(self, trajectory):
        """Summarize the episode experience for lifelong learning reflection
        Args:
            trajectory: str: The episode experience to be summarized
        """

        # Create Reflection on whole trajectories for next round trial, keep earlier messages as exemplars
        self.episode_summarization_agent.add_message(trajectory, role="user")
        subtask_summarization = call_llm_safe(self.episode_summarization_agent)
        self.episode_summarization_agent.add_message(
            subtask_summarization, role="assistant"
        )

        return subtask_summarization

    def summarize_narrative(self, trajectory):
        """Summarize the narrative experience for lifelong learning reflection
        Args:
            trajectory: str: The narrative experience to be summarized
        """
        # Create Reflection on whole trajectories for next round trial
        self.narrative_summarization_agent.add_message(trajectory, role="user")
        lifelong_learning_reflection = call_llm_safe(self.narrative_summarization_agent)

        return lifelong_learning_reflection

    def _generate_step_by_step_plan(
        self,
        observation: Dict,
        instruction: str,
        failed_subtask: Optional[Node] = None,
        completed_subtasks_list: List[Node] = [],
        remaining_subtasks_list: List[Node] = [],
    ) -> Tuple[Dict, str]:
        agent = self.grounding_agent

        # Converts a list of DAG Nodes into a natural langauge list
        def format_subtask_list(subtasks: List[Node]) -> str:
            res = ""
            for idx, node in enumerate(subtasks):
                res += f"{idx+1}. **{node.name}**:\n"
                bullets = re.split(r"(?<=[.!?;]) +", node.info)
                for bullet in bullets:
                    res += f"   - {bullet}\n"
                res += "\n"
            return res

        # Perform Retrieval only at the first planning step
        if self.turn_count == 0:

            self.search_query = self.knowledge_base.formulate_query(
                instruction, observation
            )

            most_similar_task = ""
            retrieved_experience = ""
            integrated_knowledge = ""
            # Retrieve most similar narrative (task) experience
            most_similar_task, retrieved_experience = (
                self.knowledge_base.retrieve_narrative_experience(instruction)
            )
            logger.info(
                "SIMILAR TASK EXPERIENCE: %s",
                most_similar_task + "\n" + retrieved_experience.strip(),
            )

            # Retrieve knowledge from the web if search_engine is provided
            if self.search_engine is not None:
                retrieved_knowledge = self.knowledge_base.retrieve_knowledge(
                    instruction=instruction,
                    search_query=self.search_query,
                    search_engine=self.search_engine,
                )
                logger.info("RETRIEVED KNOWLEDGE: %s", retrieved_knowledge)

                if retrieved_knowledge is not None:
                    # Fuse the retrieved knowledge and experience
                    integrated_knowledge = self.knowledge_base.knowledge_fusion(
                        observation=observation,
                        instruction=instruction,
                        web_knowledge=retrieved_knowledge,
                        similar_task=most_similar_task,
                        experience=retrieved_experience,
                    )
                    logger.info("INTEGRATED KNOWLEDGE: %s", integrated_knowledge)

            integrated_knowledge = integrated_knowledge or retrieved_experience

            # Add the integrated knowledge to the task instruction in the system prompt
            if integrated_knowledge:
                instruction += f"\nYou may refer to some retrieved knowledge if you think they are useful.{integrated_knowledge}"

            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )

        # Re-plan on failure case
        if failed_subtask:
            generator_message = (
                f"The subtask {failed_subtask} cannot be completed. Please generate a new plan for the remainder of the trajectory.\n\n"
                f"Successfully Completed Subtasks:\n{format_subtask_list(completed_subtasks_list)}\n"
            )
        # Re-plan on subtask completion case
        elif len(completed_subtasks_list) + len(remaining_subtasks_list) > 0:
            generator_message = (
                "The current trajectory and desktop state is provided. Please revise the plan for the following trajectory.\n\n"
                f"Successfully Completed Subtasks:\n{format_subtask_list(completed_subtasks_list)}\n"
                f"Future Remaining Subtasks:\n{format_subtask_list(remaining_subtasks_list)}\n"
            )
        # Initial plan case
        else:
            generator_message = "Please generate the initial plan for the task.\n"

        logger.info("GENERATOR MESSAGE: %s", generator_message)

        self.generator_agent.add_message(
            generator_message,
            image_content=observation.get("screenshot", None),
            role="user",
        )

        logger.info("GENERATING HIGH LEVEL PLAN")

        plan = call_llm_safe(self.generator_agent)
        if plan == "":
            raise Exception("Plan Generation Failed - Fix the Prompt")

        logger.info("HIGH LEVEL STEP BY STEP PLAN: %s", plan)

        self.generator_agent.add_message(plan, role="assistant")
        self.planner_history.append(plan)
        self.turn_count += 1

        # Set Cost based on GPT-4o
        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        planner_info = {
            "search_query": self.search_query,
            "goal_plan": plan,
            "num_input_tokens_plan": input_tokens,
            "num_output_tokens_plan": output_tokens,
            "goal_plan_cost": cost,
        }

        assert type(plan) == str

        return planner_info, plan

    def _generate_dag(self, instruction: str, plan: str) -> Tuple[Dict, Dag]:
        # For the re-planning case, remove the prior input since this should only translate the new plan
        self.dag_translator_agent.reset()

        # Add initial instruction and plan to the agent's message history
        self.dag_translator_agent.add_message(
            f"Instruction: {instruction}\nPlan: {plan}", role="user"
        )

        logger.info("GENERATING DAG")

        # Generate DAG
        dag_raw = call_llm_safe(self.dag_translator_agent)

        dag = parse_dag(dag_raw)

        logger.info("Generated DAG: %s", dag_raw)

        self.dag_translator_agent.add_message(dag_raw, role="assistant")

        input_tokens, output_tokens = calculate_tokens(
            self.dag_translator_agent.messages
        )

        # Set Cost based on GPT-4o
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        dag_info = {
            "dag": dag_raw,
            "num_input_tokens_dag": input_tokens,
            "num_output_tokens_dag": output_tokens,
            "dag_cost": cost,
        }

        assert type(dag) == Dag

        return dag_info, dag

    def _topological_sort(self, dag: Dag) -> List[Node]:
        """Topological sort of the DAG using DFS
        dag: Dag: Object representation of the DAG with nodes and edges
        """

        def dfs(node_name, visited, stack):
            visited[node_name] = True
            for neighbor in adj_list[node_name]:
                if not visited[neighbor]:
                    dfs(neighbor, visited, stack)
            stack.append(node_name)

        # Convert edges to adjacency list
        adj_list = defaultdict(list)
        for u, v in dag.edges:
            adj_list[u.name].append(v.name)

        visited = {node.name: False for node in dag.nodes}
        stack = []

        for node in dag.nodes:
            if not visited[node.name]:
                dfs(node.name, visited, stack)

        # Return the nodes in topologically sorted order
        sorted_nodes = [
            next(n for n in dag.nodes if n.name == name) for name in stack[::-1]
        ]
        return sorted_nodes

    def get_action_queue(
        self,
        instruction: str,
        observation: Dict,
        failed_subtask: Optional[Node] = None,
        completed_subtasks_list: List[Node] = [],
        remaining_subtasks_list: List[Node] = [],
    ):
        """Generate the action list based on the instruction
        instruction:str: Instruction for the task
        """

        planner_info, plan = self._generate_step_by_step_plan(
            observation,
            instruction,
            failed_subtask,
            completed_subtasks_list,
            remaining_subtasks_list,
        )

        # Generate the DAG
        dag_info, dag = self._generate_dag(instruction, plan)

        # Topological sort of the DAG
        action_queue = self._topological_sort(dag)

        planner_info.update(dag_info)

        return planner_info, action_queue
