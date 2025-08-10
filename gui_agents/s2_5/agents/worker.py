import logging
import textwrap
from typing import Dict, List, Tuple

from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.core.module import BaseModule
from gui_agents.s2_5.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2_5.utils.common_utils import (
    call_llm_safe,
    extract_first_agent_function,
    parse_single_code_from_string,
    sanitize_code,
    split_thinking_response,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            engine_params: Dict
                Parameters for the multimodal engine
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
        """
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.temperature = engine_params.get("temperature", 0.0)
        self.use_thinking = engine_params.get("model", "") in [
            "claude-3-7-sonnet-20250219"
        ]

        # Initialize MemOS
        mos_config = MOSConfig()
        self.memory = MOS(mos_config)
        self.user_id = "agent_s2_5_user"
        if not self.memory.user_exists(self.user_id):
            self.memory.create_user(user_id=self.user_id)

        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.turn_count = 0
        self.last_plan = ""
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        # Clear user memory for a new task
        if self.memory.user_exists(self.user_id):
            self.memory.clear_memory(user_id=self.user_id)

    # Flushing strategy dependant on model context limits
    def flush_messages(self):
        engine_type = self.engine_params.get("engine_type", "")

        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
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

    def _generate_thoughts(self, prompt, num_thoughts):
        # For simplicity, we generate thoughts in a batch.
        # A more advanced MCTS would generate them one by one.
        response = call_llm_safe(
            self.generator_agent,
            temperature=0.7, # Higher temperature for diversity
            use_thinking=self.use_thinking,
            num_responses=num_thoughts
        )
        # This is a simplification. We assume the model returns a list of thoughts.
        # In reality, we might need to parse a single string.
        try:
            thoughts = eval(response)
            if isinstance(thoughts, list):
                return thoughts
        except:
            pass
        return []


    def _evaluate_thought(self, thought, prompt_template):
        prompt = prompt_template.format(
            current_state=prompt_template.get("current_state", ""),
            task_description=prompt_template.get("task_description", ""),
            history=prompt_template.get("history", ""),
            thought_reasoning=thought.get("reasoning", ""),
            thought_action=thought.get("action", "")
        )

        eval_agent = self._create_agent(prompt)
        evaluation = call_llm_safe(eval_agent, temperature=0.0)
        try:
            return float(evaluation)
        except (ValueError, TypeError):
            return 0.0

    def generate_next_action(
        self,
        instruction: str,
        obs: Dict,
    ) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation using Graph of Thoughts.
        """
        agent = self.grounding_agent

        # 1. Retrieve relevant memories
        retrieved_memories = self.memory.search(query=instruction, user_id=self.user_id)
        memory_context = "\n".join([mem['content'] for mem_type in retrieved_memories.values() for mem in mem_type])

        if self.turn_count > 0:
            generator_message = f"Here is a summary of relevant past actions:\n{memory_context}\n"
        else:
            generator_message = "The initial screen is provided. No action has been taken yet."

        # Load the task into the system prompt
        if self.turn_count == 0:
            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace("TASK_DESCRIPTION", instruction)
            )

        # 2. Get reflection (if enabled)
        reflection = None
        if self.enable_reflection and self.turn_count > 0:
            self.reflection_agent.add_message(text_content=self.last_plan, image_content=obs["screenshot"], role="user")
            full_reflection = call_llm_safe(self.reflection_agent, temperature=self.temperature, use_thinking=self.use_thinking)
            reflection, _ = split_thinking_response(full_reflection)
            self.reflections.append(reflection)
            generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"
            logger.info("REFLECTION: %s", reflection)

        generator_message += f"\nCurrent Text Buffer = [{','.join(agent.notes)}]\n"

        # 3. Graph of Thoughts / MCTS - Simplified
        # 3.1. Generate multiple thoughts
        num_thoughts_to_generate = 3
        generation_prompt = PROCEDURAL_MEMORY.GENERATE_THOUGHTS_PROMPT.format(
            current_state=generator_message,
            task_description=instruction,
            history=memory_context,
            num_thoughts=num_thoughts_to_generate
        )
        self.generator_agent.add_message(generation_prompt, image_content=obs["screenshot"], role="user")

        raw_thoughts = call_llm_safe(self.generator_agent, temperature=0.5)

        try:
            thoughts = eval(raw_thoughts)
            if not isinstance(thoughts, list): thoughts = []
        except:
            thoughts = []

        # 3.2. Evaluate thoughts
        best_thought = None
        max_score = -1.0

        evaluation_prompt_template = {
            "current_state": generator_message,
            "task_description": instruction,
            "history": memory_context
        }

        for thought in thoughts:
            score = self._evaluate_thought(thought, evaluation_prompt_template)
            thought['score'] = score
            if score > max_score:
                max_score = score
                best_thought = thought

        if not best_thought: # Fallback to original method if GoT fails
            logger.warning("GoT failed to produce a valid thought, falling back to single-path generation.")
            self.generator_agent.add_message(generator_message, image_content=obs["screenshot"], role="user")
            plan = call_llm_safe(self.generator_agent, temperature=self.temperature)
        else:
            plan = f"(Reasoning)\n{best_thought['reasoning']}\n(Grounded Action)\n```python\n{best_thought['action']}\n```"

        # 4. Process and execute the best plan
        # Add the current turn to memory
        user_msg_for_memory = {"role": "user", "content": generator_message}
        assistant_msg_for_memory = {"role": "assistant", "content": plan}
        self.memory.add(messages=[user_msg_for_memory, assistant_msg_for_memory], user_id=self.user_id)
        self.last_plan = plan

        logger.info("FULL PLAN:\n %s", plan)
        self.generator_agent.add_message(plan, role="assistant")

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
            "full_plan": plan, # Simplified, full plan is not available in GoT
            "executor_plan": plan,
            "plan_thoughts": thoughts, # Include all generated thoughts for analysis
            "plan_code": plan_code,
            "reflection": reflection,
            "reflection_thoughts": None, # Not available in this simplified version
            "memory_context": memory_context
        }
        self.turn_count += 1

        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()

        return executor_info, [exec_code]
