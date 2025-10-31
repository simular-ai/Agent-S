from functools import partial
import logging
import textwrap
from typing import Dict, List, Tuple

from gui_agents.s3.agents.grounding import ACI
from gui_agents.s3.core.module import BaseModule
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from gui_agents.s3.utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            worker_engine_params: Dict
                Parameters for the worker agent
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
        """
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        self.use_thinking = worker_engine_params.get("model", "") in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
        ]
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection

        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        # Hide code agent action entirely if no env/controller is available
        if not getattr(self.grounding_agent, "env", None) or not getattr(
            getattr(self.grounding_agent, "env", None), "controller", None
        ):
            skipped_actions.append("call_code_agent")

        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []

    def flush_messages(self):
        """Flush messages based on the model's context limits.

        This method ensures that the agent's message history does not exceed the maximum trajectory length.

        Side Effects:
            - Modifies the messages of generator, reflection, and bon_judge agents to fit within the context limits.
        """
        engine_type = self.engine_params.get("engine_type", "")

        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
                if agent is None:
                    continue
                # keep latest k images
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]

        # Flush strategy for non-long-context models: drop full turns
        else:
            # generator msgs are alternating [user, assistant], so 2 per round
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
            # reflector msgs are all [(user text, user image)], so 1 per round
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)

    def _generate_reflection(self, instruction: str, obs: Dict) -> Tuple[str, str]:
        """
        Generate a reflection based on the current observation and instruction.

        Args:
            instruction (str): The task instruction.
            obs (Dict): The current observation containing the screenshot.

        Returns:
            Optional[str, str]: The generated reflection text and thoughts, if any (turn_count > 0).

        Side Effects:
            - Updates reflection agent's history
            - Generates reflection response with API call
        """
        reflection = None
        reflection_thoughts = None
        if self.enable_reflection:
            # Load the initial message
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
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
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1],
                    image_content=obs["screenshot"],
                    role="user",
                )
                full_reflection = call_llm_safe(
                    self.reflection_agent,
                    temperature=self.temperature,
                    use_thinking=self.use_thinking,
                )
                reflection, reflection_thoughts = split_thinking_response(
                    full_reflection
                )
                self.reflections.append(reflection)
                logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
                logger.info("REFLECTION: %s", reflection)
        return reflection, reflection_thoughts

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """

        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)

        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )

        # Load the task into the system prompt
        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)

        # Get the per-step reflection
        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)
        if reflection:
            generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"

        # Get the grounding agent's knowledge base buffer
        generator_message += (
            f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        )

        # Add code agent result from previous step if available (from full task or subtask execution)
        if (
            hasattr(self.grounding_agent, "last_code_agent_result")
            and self.grounding_agent.last_code_agent_result is not None
        ):
            code_result = self.grounding_agent.last_code_agent_result
            generator_message += f"\nCODE AGENT RESULT:\n"
            generator_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            generator_message += f"Steps Completed: {code_result['steps_executed']}\n"
            generator_message += f"Max Steps: {code_result['budget']}\n"
            generator_message += (
                f"Completion Reason: {code_result['completion_reason']}\n"
            )
            generator_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                generator_message += f"Execution History:\n"
                for i, step in enumerate(code_result["execution_history"]):
                    action = step["action"]
                    # Format code snippets with proper backticks
                    if "```python" in action:
                        # Extract Python code and format it
                        code_start = action.find("```python") + 9
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            python_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```python\n{python_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    elif "```bash" in action:
                        # Extract Bash code and format it
                        code_start = action.find("```bash") + 7
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            bash_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```bash\n{bash_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    else:
                        generator_message += f"Step {i+1}: \n{action}\n"
            generator_message += "\n"

            # Log the code agent result section for debugging (truncated execution history)
            log_message = f"\nCODE AGENT RESULT:\n"
            log_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            log_message += f"Steps Completed: {code_result['steps_executed']}\n"
            log_message += f"Max Steps: {code_result['budget']}\n"
            log_message += f"Completion Reason: {code_result['completion_reason']}\n"
            log_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                log_message += f"Execution History (truncated):\n"
                # Only log first 3 steps and last 2 steps to keep logs manageable
                total_steps = len(code_result["execution_history"])
                for i, step in enumerate(code_result["execution_history"]):
                    if i < 3 or i >= total_steps - 2:  # First 3 and last 2 steps
                        action = step["action"]
                        if "```python" in action:
                            code_start = action.find("```python") + 9
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                python_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```python\n{python_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        elif "```bash" in action:
                            code_start = action.find("```bash") + 7
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                bash_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```bash\n{bash_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        else:
                            log_message += f"Step {i+1}: {action}\n"
                    elif i == 3 and total_steps > 5:
                        log_message += f"... (truncated {total_steps - 5} steps) ...\n"

            logger.info(
                f"WORKER_CODE_AGENT_RESULT_SECTION - Step {self.turn_count + 1}: Code agent result added to generator message:\n{log_message}"
            )

            # Reset the code agent result after adding it to context
            self.grounding_agent.last_code_agent_result = None

        # Finalize the generator message
        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )

        # Generate the plan and next action
        format_checkers = [
            SINGLE_ACTION_FORMATTER,
            partial(CODE_VALID_FORMATTER, self.grounding_agent, obs),
        ]
        plan = call_llm_formatted(
            self.generator_agent,
            format_checkers,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        self.worker_history.append(plan)
        self.generator_agent.add_message(plan, role="assistant")
        logger.info("PLAN:\n %s", plan)

        # Extract the next action from the plan
        plan_code = parse_code_from_string(plan)
        try:
            assert plan_code, "Plan code should not be empty"
            exec_code = create_pyautogui_code(self.grounding_agent, plan_code, obs)
        except Exception as e:
            logger.error(
                f"Could not evaluate the following plan code:\n{plan_code}\nError: {e}"
            )
            exec_code = self.grounding_agent.wait(
                1.333
            )  # Skip a turn if the code cannot be evaluated

        executor_info = {
            "plan": plan,
            "plan_code": plan_code,
            "exec_code": exec_code,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "code_agent_output": (
                self.grounding_agent.last_code_agent_result
                if hasattr(self.grounding_agent, "last_code_agent_result")
                and self.grounding_agent.last_code_agent_result is not None
                else None
            ),
        }
        self.turn_count += 1
        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()
        return executor_info, [exec_code]
