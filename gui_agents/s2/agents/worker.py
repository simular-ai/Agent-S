import logging
import re
import textwrap
from typing import Dict, List, Tuple
import platform

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.core.module import BaseModule
from gui_agents.s2.core.knowledge import KnowledgeBase
from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.utils.common_utils import (
    Node,
    calculate_tokens,
    call_llm_safe,
    parse_single_code_from_string,
    sanitize_code,
    extract_first_agent_function,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        local_kb_path: str,
        embedding_engine,
        platform: str = platform.system().lower(),
        enable_reflection: bool = True,
        use_subtask_experience: bool = True,
    ):
        """
        Worker receives a subtask list and active subtask and generates the next action for the to execute.
        Args:
            engine_params: Dict
                Parameters for the multimodal engine
            grounding_agent: Agent
                The grounding agent to use
            local_kb_path: str
                Path to knowledge base
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            enable_reflection: bool
                Whether to enable reflection
            use_subtask_experience: bool
                Whether to use subtask experience
        """
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent
        self.local_kb_path = local_kb_path
        self.embedding_engine = embedding_engine
        self.enable_reflection = enable_reflection
        self.use_subtask_experience = use_subtask_experience
        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        sys_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.knowledge_base = KnowledgeBase(
            embedding_engine=self.embedding_engine,
            local_kb_path=self.local_kb_path,
            platform=self.platform,
            engine_params=self.engine_params,
        )

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        self.planner_history = []
        self.max_trajector_length = 8

    def flush_messages(self):
        # generator msgs are alternating [user, assistant], so 2 per round
        if len(self.generator_agent.messages) > 2 * self.max_trajector_length + 1:
            self.generator_agent.remove_message_at(1)
            self.generator_agent.remove_message_at(1)
        # reflector msgs are all [(user text, user image)], so 1 per round
        if len(self.reflection_agent.messages) > self.max_trajector_length + 1:
            self.reflection_agent.remove_message_at(1)

    def generate_next_action(
        self,
        instruction: str,
        search_query: str,
        subtask: str,
        subtask_info: Dict,
        future_tasks: List[Node],
        done_task: List[Node],
        obs: Dict,
    ) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """
        # Provide the top_app to the Grounding Agent to remove all other applications from the tree. At t=0, top_app is None
        agent = self.grounding_agent

        # Get RAG knowledge, only update system message at t=0
        if self.turn_count == 0:
            if self.use_subtask_experience:
                subtask_query_key = (
                    "Task:\n"
                    + search_query
                    + "\n\nSubtask: "
                    + subtask
                    + "\nSubtask Instruction: "
                    + subtask_info
                )
                retrieved_similar_subtask, retrieved_subtask_experience = (
                    self.knowledge_base.retrieve_episodic_experience(subtask_query_key)
                )

                # Dirty fix to replace id with element description during subtask retrieval
                pattern = r"\(\d+"
                retrieved_subtask_experience = re.sub(
                    pattern, "(element_description", retrieved_subtask_experience
                )
                retrieved_subtask_experience = retrieved_subtask_experience.replace(
                    "_id", "_description"
                )

                logger.info(
                    "SIMILAR SUBTASK EXPERIENCE: %s",
                    retrieved_similar_subtask
                    + "\n"
                    + retrieved_subtask_experience.strip(),
                )
                instruction += "\nYou may refer to some similar subtask experience if you think they are useful. {}".format(
                    retrieved_similar_subtask + "\n" + retrieved_subtask_experience
                )

            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "SUBTASK_DESCRIPTION", subtask
                )
                .replace("TASK_DESCRIPTION", instruction)
                .replace("FUTURE_TASKS", ", ".join([f.name for f in future_tasks]))
                .replace("DONE_TASKS", ",".join(d.name for d in done_task))
            )

        # Reflection generation does not add its own response, it only gets the trajectory
        reflection = None
        if self.enable_reflection:
            # Load the initial subtask info
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Subtask Description: {subtask}
                    Subtask Information: {subtask_info}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=obs["screenshot"],
                    role="user",
                )
            # Load the latest action
            else:
                text_content = self.clean_worker_generation_for_reflection(
                    self.planner_history[-1]
                )
                self.reflection_agent.add_message(
                    text_content=text_content,
                    image_content=obs["screenshot"],
                    role="user",
                )
                reflection = call_llm_safe(self.reflection_agent)
                self.reflections.append(reflection)
                logger.info("REFLECTION: %s", reflection)

        generator_message = (
            f"\nYou may use this reflection on the previous action and overall trajectory: {reflection}\n"
            if reflection and self.turn_count > 0
            else ""
        ) + f"Text Buffer = [{','.join(agent.notes)}]."

        # Only provide subinfo in the very first message to avoid over influence and redundancy
        if self.turn_count == 0:
            generator_message += f"Remember only complete the subtask: {subtask}\n"
            generator_message += f"You can use this extra information for completing the current subtask: {subtask_info}.\n"

        # logger.info("GENERATOR MESSAGE: %s", generator_message)

        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )

        plan = call_llm_safe(self.generator_agent)
        self.planner_history.append(plan)
        logger.info("PLAN: %s", plan)
        self.generator_agent.add_message(plan, role="assistant")

        # Calculate input/output tokens and gpt-4o cost
        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        self.cost_this_turn += cost
        logger.info("EXECTUOR COST: %s", self.cost_this_turn)

        # Use the DescriptionBasedACI to convert agent_action("desc") into agent_action([x, y])
        try:
            agent.assign_coordinates(plan, obs)
            plan_code = parse_single_code_from_string(plan.split("Grounded Action")[-1])
            plan_code = sanitize_code(plan_code)
            plan_code = extract_first_agent_function(plan_code)
            exec_code = eval(plan_code)
        except Exception as e:
            logger.error("Error in parsing plan code: %s", e)
            plan_code = "agent.wait(1.0)"
            exec_code = eval(plan_code)

        executor_info = {
            "current_subtask": subtask,
            "current_subtask_info": subtask_info,
            "executor_plan": plan,
            "plan_code": plan_code,
            "reflection": reflection,
            "num_input_tokens_executor": input_tokens,
            "num_output_tokens_executor": output_tokens,
        }
        self.turn_count += 1

        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()

        return executor_info, [exec_code]

    # Removes the previous action verification, and removes any extraneous grounded actions
    def clean_worker_generation_for_reflection(self, worker_generation: str) -> str:
        # Remove the previous action verification
        res = worker_generation[worker_generation.find("(Screenshot Analysis)") :]
        action = extract_first_agent_function(worker_generation)
        # Cut off extra grounded actions
        res = res[: res.find("(Grounded Action)")]
        res += f"(Grounded Action)\n```python\n{action}\n```\n"
        return res
