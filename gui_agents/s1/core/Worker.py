import logging
import os
import re
from typing import Dict, List, Tuple
import platform

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.core.BaseModule import BaseModule
from gui_agents.s1.core.Knowledge import KnowledgeBase
from gui_agents.s1.core.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.s1.utils import common_utils
from gui_agents.s1.utils.common_utils import Node, calculate_tokens, call_llm_safe

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        local_kb_path: str,
        platform: str = platform.system().lower(),
        search_engine: str = "perplexica",
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
            search_engine: str
                The search engine to use
            enable_reflection: bool
                Whether to enable reflection
            use_subtask_experience: bool
                Whether to use subtask experience
        """
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent
        self.local_kb_path = local_kb_path
        self.enable_reflection = enable_reflection
        self.search_engine = search_engine
        self.use_subtask_experience = use_subtask_experience
        self.reset()

    def flush_messages(self, n):
        # After every max_trajectory_length trajectories, remove messages from the start except the system prompt
        for agent in [self.generator_agent]:
            if len(agent.messages) > 2 * n + 1:
                # Remove the user message and assistant message, both are 1 because the elements will move back after 1 pop
                agent.remove_message_at(1)
                agent.remove_message_at(1)

    def reset(self):
        self.generator_agent = self._create_agent(
            PROCEDURAL_MEMORY.construct_worker_procedural_memory(
                type(self.grounding_agent)
            ).replace("CURRENT_OS", self.platform)
        )
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.knowledge_base = KnowledgeBase(
            local_kb_path=self.local_kb_path,
            platform=self.platform,
            engine_params=self.engine_params,
        )

        self.turn_count = 0
        self.planner_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.tree_inputs = []
        self.screenshot_inputs = []

    # TODO: Experimental
    def remove_ids_from_history(self):
        for message in self.generator_agent.messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "text":
                        # Regex pattern to match lines that start with a number followed by spaces and remove the number
                        pattern = r"^\d+\s+"

                        # Apply the regex substitution on each line
                        processed_lines = [
                            re.sub(pattern, "", line)
                            for line in content["text"].splitlines()
                        ]

                        # Join the processed lines back into a single string
                        result = "\n".join(processed_lines)

                        result = result.replace("id\t", "")

                        # replace message content
                        content["text"] = result

    def generate_next_action(
        self,
        instruction: str,
        search_query: str,
        subtask: str,
        subtask_info: str,
        future_tasks: List[Node],
        done_task: List[Node],
        obs: Dict,
    ) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """
        # Provide the top_app to the Grounding Agent to remove all other applications from the tree. At t=0, top_app is None
        agent = self.grounding_agent

        self.active_apps = agent.get_active_apps(obs)

        # Get RAG knowledge, only update system message at t=0
        if self.turn_count == 0:
            # TODO: uncomment and fix for subtask level RAG
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

        # Clear older messages - we keep full context. if you want to keep only the last n messages, you can use the flush_messages function
        # self.flush_messages(3) # flushes generator messages

        # Reflection generation
        reflection = None
        if self.enable_reflection and self.turn_count > 0:
            # TODO: reuse planner history
            self.reflection_agent.add_message(
                "Task Description: "
                + subtask
                + " Instruction: "
                + subtask_info
                + "\n"
                + "Current Trajectory: "
                + "\n\n".join(self.planner_history)
                + "\n"
            )
            reflection = call_llm_safe(self.reflection_agent)
            self.reflections.append(reflection)
            self.reflection_agent.add_message(reflection)

            logger.info("REFLECTION: %s", reflection)

        # Plan Generation
        tree_input = agent.linearize_and_annotate_tree(obs)

        self.remove_ids_from_history()

        # Bash terminal message.
        generator_message = (
            (
                f"\nYou may use the reflection on the previous trajectory: {reflection}\n"
                if reflection
                else ""
            )
            + f"Accessibility Tree: {tree_input}\n"
            f"Text Buffer = [{','.join(agent.notes)}]. "
            f"The current open applications are {agent.get_active_apps(obs)} and the active app is {agent.get_top_app(obs)}.\n"
        )

        print("ACTIVE APP IS: ", agent.get_top_app(obs))
        # Only provide subinfo in the very first message to avoid over influence and redundancy
        if self.turn_count == 0:
            generator_message += f"Remeber only complete the subtask: {subtask}\n"
            generator_message += f"You can use this extra information for completing the current subtask: {subtask_info}.\n"

        logger.info("GENERATOR MESSAGE: %s", generator_message)

        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"]
        )

        plan = call_llm_safe(self.generator_agent)
        self.planner_history.append(plan)
        logger.info("PLAN: %s", plan)

        self.generator_agent.add_message(plan)

        # Calculate input and output tokens
        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)

        # Set Cost based on GPT-4o
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        self.cost_this_turn += cost
        logger.info("EXECTUOR COST: %s", self.cost_this_turn)

        # Extract code block from the plan
        plan_code = common_utils.parse_single_code_from_string(
            plan.split("Grounded Action")[-1]
        )
        plan_code = common_utils.sanitize_code(plan_code)
        plan_code = common_utils.extract_first_agent_function(plan_code)
        exec_code = eval(plan_code)

        # If agent selects an element that was out of range, it should not be executed just send a WAIT command.
        # TODO: should provide this as code feedback to the agent?
        if agent.index_out_of_range_flag:
            plan_code = "agent.wait(1.0)"
            exec_code = eval(plan_code)
            agent.index_out_of_range_flag = False

        executor_info = {
            "current_subtask": subtask,
            "current_subtask_info": subtask_info,
            "executor_plan": plan,
            "linearized_accessibility_tree": tree_input,
            "plan_code": plan_code,
            "reflection": reflection,
            "num_input_tokens_executor": input_tokens,
            "num_output_tokens_executor": output_tokens,
            "executor_cost": cost,
        }
        self.turn_count += 1

        self.tree_inputs.append(tree_input)
        self.screenshot_inputs.append(obs["screenshot"])

        return executor_info, [exec_code]
