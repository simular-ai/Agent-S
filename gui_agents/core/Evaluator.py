import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from desktop_env.desktop_env import DesktopEnv
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from gui_agents import osworld_utils
from gui_agents.MultimodalAgent import LMMAgent
from gui_agents.MultimodalEngine import OpenAIEmbeddingEngine
from gui_agents.osworld.GroundingAgent import GroundingAgent
from gui_agents.osworld_utils import Dag, Node
from gui_agents.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.query_perplexica import query_to_perplexica


class Evaluator:
    def __init__(self, instruction, engine_params: Dict, script_check: bool = False):
        self.instruction = instruction
        self.engine_params = engine_params
        self.reset()

    def reset(self):
        self.state_evaluator = LMMAgent(self.engine_params)
        self.state_evaluator_system_prompt = PROCEDURAL_MEMORY.STATE_EVALUATOR_SYSTEM_PROMPT
        self.state_evaluator.add_system_prompt(
            self.state_evaluator_system_prompt.replace("TASK_DESCRIPTION", self.instruction))

        self.obs_evaluator = LMMAgent(self.engine_params)
        self.obs_evaluator_system_prompt = PROCEDURAL_MEMORY.OBS_EVALUATOR_SYSTEM_PROMPT
        self.obs_evaluator.add_system_prompt(
            self.obs_evaluator_system_prompt.replace("TASK_DESCRIPTION", self.instruction))

    def state_evaluate(self, input, input_img, plan_codes, env: DesktopEnv = None):
        init_obs, last_obs = input[0], input[-1]
        init_obs_img, last_obs_img = input_img[0], input_img[-1]

        input_message_1 = f"""
                        The accessibility tree at the first step:{init_obs}, and the screenshot at the first step: \n
                        """
        input_message_2 = f"""
                        The accessibility tree at the last step:{last_obs}, and the screenshot at the last step: \n
                        """
        input_message_3 = '\nThe whole actions performed by the digital agent:\n' + \
            '\n'.join(plan_codes)
        self.state_evaluator.add_message(
            text_content=input_message_1, image_content=init_obs_img, role="user")
        self.state_evaluator.add_message(
            text_content=input_message_2, image_content=last_obs_img, role="user")
        self.state_evaluator.add_message(
            text_content=input_message_3, role="user")
        script_response = self.state_evaluator.get_response()
        print(
            f"The evaluation result of current task: {self.instruction}:\n {script_response}")
        self.state_evaluator.add_message(script_response)
        try:
            script_response = script_response.split('Judgment:')[1]
        except:
            script_response = script_response

        if 'Yes' in script_response:
            eval_result = 1.0
            with open('script_result.txt', 'a', encoding='utf-8') as f:
                f.write(
                    f"The task {self.instruction} generate the eval script: \n{script_response}\n")
            return eval_result
        elif 'No' in script_response:
            eval_result = 0.0
            with open('script_result.txt', 'a', encoding='utf-8') as f:
                f.write(
                    f"The task {self.instruction} generate the eval script: \n{script_response}\n")
            return eval_result
        else:
            script = osworld_utils.parse_single_code_from_string(
                script_response)
            script_run_output = env.controller.execute_python_command(script)
            matching_message = f"""
                                The output after executing the script is: {script_run_output}, now please do the subsequent task to judge the completeness of the task based on the script's result and the task information(Like accessibility trees, screenshots, whole actions).
                                The Script and the task regarding information is also aforementioned. Provide your analysis and put the judgment at the end of the response in this format: Judgment: Yes/No
                                """
            print(matching_message)
            self.state_evaluator.add_message(matching_message)
            matching_response = self.state_evaluator.get_response()
            print(
                f"The matching result of current task{self.instruction}:\n {matching_response}")
            with open('script_result.txt', 'a', encoding='utf-8') as f:
                f.write(f"The task: {self.instruction} generate the eval script:\n {script_response} \n, and the script_run_output is {script_run_output} \n, the matching response is {matching_response} \n\n")
            try:
                if 'Yes' in matching_response.split('Judgment:')[1]:
                    eval_result = 1.0
                else:
                    eval_result = 0.0
            except:
                if 'Yes' in matching_response:
                    eval_result = 1.0
                else:
                    eval_result = 0.0
            return eval_result

    def obs_evaluate(self, instruction: str, input: List[str], input_img: List[bytes], plan_codes: List[str]):

        self.obs_evaluator.add_system_prompt(
            self.obs_evaluator_system_prompt.replace("TASK_DESCRIPTION", instruction))

        init_obs, last_obs = input[0], input[-1]
        init_obs_img, last_obs_img = input_img[0], input_img[-1]

        input_message_1 = f"""
                        The accessibility tree at the first step:{init_obs}, and the screenshot at the first step: \n
                        """
        input_message_2 = f"""
                        The accessibility tree at the last step:{last_obs}, and the screenshot at the last step: \n
                        """
        input_message_3 = '\nThe whole actions performed by the digital agent:\n' + \
            '\n'.join(plan_codes)

        input_message = input_message_1 + input_message_2 + input_message_3

        self.obs_evaluator.add_message(input_message, image_content=[
                                       init_obs_img, last_obs_img])
        response = call_llm_safe(self.obs_evaluator)
        logger.info(
            f"The evaluation result of current subtask: {instruction}:\n {response}")

        # TODO: Expand coverage
        def check_judgment(response):
            # Improved regex pattern to match "Judgment: Yes" or "Judgment: No" at the end of the response, allowing extra spaces or newlines
            pattern = r"Judgment:\s*(yes|no)\s*$"

            # Search for the pattern in the response, case insensitive
            match = re.search(pattern, response.strip(),
                              re.IGNORECASE | re.MULTILINE)
            eval_result = 0
            if match:
                # Normalize the judgment (capitalize the first letter)
                judgment = match.group(1).capitalize()
                # Set eval_result based on the judgment
                eval_result = 1 if judgment == "Yes" else 0

            return eval_result

        try:
            eval_result = check_judgment(response)
        except:
            logger.error(
                "Failed to extract judgment from the response. Defaulting to 0.")
            eval_result = 0

        input_tokens, output_tokens = calculate_tokens(
            self.obs_evaluator.messages)

        # Set Cost based on GPT-4o
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        logger.info("EVALUATION COST: %s", cost)

        evaluator_info = {
            'obs_evaluator_response': response,
            'num_input_tokens_evaluator': input_tokens,
            'num_output_tokens_evaluator': output_tokens,
            'evaluator_cost': cost
        }

        return evaluator_info, eval_result, response
