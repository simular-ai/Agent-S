from agent_s.osworld.GroundingAgent import GroundingAgent as OSWorldGroundingAgent
from agent_s.windowsagentarena.GroundingAgent import GroundingAgent as WindowsAgentArenaGroundingAgent

import platform
import time 

import pyautogui
import io 

try:
    if platform.system() == 'Darwin':
        platform_name = 'macos'
        from openaci.macos.Grounding import GroundingAgent as OpenACIGroundingAgent
        from openaci.macos.Grounding import OpenACIUIElement
    elif platform.system() == 'Linux':
        platform_name = 'ubuntu'
        from openaci.ubuntu.Grounding import GroundingAgent as OpenACIGroundingAgent
        from openaci.ubuntu.UIElement import OpenACIUIElement
except ImportError:
    print(
        "The OpenACI package is not installed. To use the OpenACI grounding agent for Agent S, please install the OpenACI package."
        "\n\nIf you're running experiments on OSWorld or WindowsAgentArena, you can proceed without it."
        "\n\nTo install OpenACI, visit the official repo at: https://github.com/simular-ai/OpenACI"
    )

import os
import re
import logging
from agent_s.DAGPlanner import Planner
from agent_s.TaskExecutor import Executor
from typing import List, Dict, Tuple
logger = logging.getLogger("desktopenv.agent")

# Get the directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
file_path = os.path.join(working_dir, "kb", "formulate_query.json")

NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


class GraphSearchAgent:
    def __init__(self,
                 engine_params,
                 experiment_type='osworld',
                 platform="ubuntu",
                 max_tokens=1500,
                 top_p=0.9,
                 temperature=0.5,
                 action_space="pyautogui",
                 observation_type="a11y_tree",
                 max_trajectory_length=10,
                 a11y_tree_max_tokens=10000,
                 enable_reflection=True,
                 engine="perplexica",
                 vm_version="old"):

        # resets the agent by initializing submodules
        self.experiment_type = experiment_type
        self.engine_params = engine_params
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = 5
        self.engine = engine
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.vm_version = vm_version
        self.self_eval = True

        self.reset()

    def reset(self):
        # Initialize Agents
        if self.experiment_type == 'osworld':
            self.grounding_agent = OSWorldGroundingAgent(vm_version=self.vm_version)
        elif self.experiment_type == 'windowsagentarena':
            self.grounding_agent = WindowsAgentArenaGroundingAgent(vm_version='windows')
        elif self.experiment_type == 'openaci':
            self.grounding_agent = OpenACIGroundingAgent()
        else:
            raise ValueError(f"Invalid experiment type: {self.experiment_type}")

        self.planner = Planner(self.engine_params, self.grounding_agent)
        self.executor = Executor(self.engine_params, self.grounding_agent)
        self.replan = True
        self.get_next_subtask = True
        self.step_count = 0
        self.turn_count = 0
        self.failure_feedback = ''
        self.send_action = False
        self.done_tasks = []
        self.subtask_info = None

    def reset_executor_evaluator_states(self):
        self.executor.reset()
        self.step_count = 0

    def predict(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """

        # Initialize the three info dictionaries
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            'obs_evaluator_response': '',
            'num_input_tokens_evaluator': 0,
            'num_output_tokens_evaluator': 0,
            'evaluator_cost': 0.
        }
        actions = []

        # If the DONE response by the executor is for a subtask, then the agent should continue with the next subtask without sending the action to the environment
        while not self.send_action:
            self.subtask_status = 'In'
            # if replan is true, generate a new plan. True at start, then true again after a failed plan
            if self.replan:
                logger.info("(RE)PLANNING...")
                # failure feedback is the reason for the failure of the previous plan
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    initial_observation=obs,
                    failure_feedback=self.failure_feedback,
                    replan=False if self.turn_count == 0 else self.replan)

                self.replan = False
                if 'search_query' in planner_info:
                    self.search_query = planner_info['search_query']
                else:
                    self.search_query = ""

            # use the exectuor to complete the topmost subtask
            if self.get_next_subtask:
                logger.info("GETTING NEXT SUBTASK...")
                self.current_subtask = self.subtasks.pop(0)
                logger.info(f"NEXT SUBTASK: {self.current_subtask}")
                self.get_next_subtask = False
                self.subtask_status = 'Start'

            # get the next action from the executor
            executor_info, actions = self.executor.generate_next_action(
                instruction=instruction,
                search_query=self.search_query,
                subtask=self.current_subtask.name,
                subtask_info=self.current_subtask.info,
                future_tasks=self.subtasks,
                done_task=self.done_tasks,
                obs=obs)

            self.step_count += 1

            # set the send_action flag to True if the executor returns an action
            self.send_action = True
            if 'FAIL' in actions:
                self.replan = True
                # set the failure feedback to the evaluator feedback
                self.failure_feedback = f"Completed subtasks: {self.done_tasks}. The subtask {self.current_subtask} cannot be completed. Please try another approach. {executor_info['plan_code']}. Please replan."
                self.get_next_subtask = True

                # reset the step count, executor, and evaluator
                self.reset_executor_evaluator_states()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.send_action = False

            elif 'DONE' in actions:
                self.replan = False
                self.done_tasks.append(self.current_subtask)
                self.get_next_subtask = True
                if self.subtasks:
                    self.send_action = False
                self.subtask_status = 'Done'

                self.reset_executor_evaluator_states()

            self.turn_count += 1
        # reset the send_action flag for next iteration
        self.send_action = False

        # concatenate the three info dictionaries
        info = {**{k: v for d in [planner_info or {}, executor_info or {},
                                  evaluator_info or {}] for k, v in d.items()}}
        info.update({
        'subtask': self.current_subtask.name,
        'subtask_info': self.current_subtask.info,
        'subtask_status': self.subtask_status})

        if self.self_eval:
            curr_atree = self.grounding_agent.linearize_and_annotate_tree(obs)
            curr_img = obs['screenshot']
            return info, (curr_atree, curr_img), actions
        else:
            return info, actions
    
    def run(self, instruction: str):
        obs = {}
        traj = 'Task:\n' + instruction
        subtask_traj = ""
        for _ in range(15):
            obs['accessibility_tree'] = OpenACIUIElement.systemWideElement()
                
            # Get screen shot using pyautogui.
            # Take a screenshot
            screenshot = pyautogui.screenshot()

            # Save the screenshot to a BytesIO object
            buffered = io.BytesIO()
            screenshot.save(buffered, format="PNG")

            # Get the byte value of the screenshot
            screenshot_bytes = buffered.getvalue()
            # Convert to base64 string.
            obs['screenshot'] = screenshot_bytes 

            info, code = self.predict(instruction=instruction, obs=obs)

            if 'done' in code[0].lower() or 'fail' in code[0].lower():
                if platform.system() == 'Darwin':
                    os.system(f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\'')
                elif platform.system() == 'Linux':
                    os.system(f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100')
                
                self.update_narrative_memory(traj)
                break 
        
            if _ == 3:
                self.update_narrative_memory(traj)
            
            if 'next' in code[0].lower():
                continue

            if 'wait' in code[0].lower():
                time.sleep(5)
                continue

            else:
                time.sleep(1.)
                print("EXECUTING CODE:", code[0])
                exec(code[0])
                
                time.sleep(1.)
                
                # Update task and subtask trajectories and optionally the episodic memory
                traj += '\n\nReflection:\n' + str(info['reflection']) + '\n\n----------------------\n\nPlan:\n' + info['executor_plan']
                subtask_traj = self.update_episodic_memory(info, subtask_traj)
