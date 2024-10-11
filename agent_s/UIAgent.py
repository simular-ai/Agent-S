import time
import xml.etree.ElementTree as ET
from agent_s.ProceduralMemory import PROCEDURAL_MEMORY
from agent_s.agent_s.osworld.GroundingAgent import GroundingAgent
from agent_s.MultimodalEngine import OpenAIEmbeddingEngine
import numpy as np
import json
import io
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple
from agent_s.MultimodalAgent import LMMAgent
import logging
from typing import Dict, List
from agent_s import osworld_utils
from agent_s.query_perplexica import query_to_perplexica

logger = logging.getLogger("desktopenv.agent")

# Get the directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
file_path = os.path.join(working_dir, "kb", "formulate_query.json")

NUM_IMAGE_TOKEN=1105 # Value set of screen of size 1920x1080 for openai vision

# TODO: Rename this class and unify with grounding variations and planning variations
class IDBasedGroundingUIAgent:
    def __init__(self,
                 engine_params,
                 platform="ubuntu",
                 max_tokens=1500,
                 top_p=0.9,
                 temperature=0.5,
                 action_space="pyautogui",
                 observation_type="a11y_tree",
                 max_trajectory_length=3,
                 a11y_tree_max_tokens=10000,
                 enable_reflection=True,
                 engine="perplexica"):

        # Initialize Agents
        self.rag_agent = LMMAgent(engine_params)
        self.planning_agent = LMMAgent(engine_params)
        self.reflection_agent = LMMAgent(engine_params)
        self.lifelong_learning_agent = LMMAgent(engine_params)

        # Initialize Embedding Engine
        self.embedding_engine = OpenAIEmbeddingEngine()

        # Set parameters
        self.enable_reflection = enable_reflection
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.engine = engine

        # Initialize variables
        self.plans = []
        self.actions = []
        self.inputs = []
        self.messages = []
        self.feedbacks = []
        self.reflections = []

        if observation_type == "screenshot_a11y_tree":
            self.planning_module_system_prompt = PROCEDURAL_MEMORY.PLANNING_AGENT_ID_BASED_GROUNDING_FIXED_ACTION_SPACE_REACT_STYLE_WITH_SCREENSHOT
            logger.info("Using screenshot_a11y_tree Prompt for Planning Module")
        else:
            self.planning_module_system_prompt = PROCEDURAL_MEMORY.PLANNING_AGENT_ID_BASED_GROUNDING_FIXED_ACTION_SPACE_REACT_STYLE
            logger.info("Using a11y_tree Prompt for Planning Module")
        self.feedback_module_system_prompt = None
        self.action_module_system_prompt = None
        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT
        self.visual_grounding_verification_module_system_prompt = PROCEDURAL_MEMORY.VISUAL_GROUNDING_VERIFICATION
        self.reflection_module_system_prompt = PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        self.lifelong_learning_system_prompt = PROCEDURAL_MEMORY.LIFELONG_LEARNING_REFLECTION
        self.prev_atree = None
        self.turn_count = None
        self.cost_this_turn = 0

        self.active_apps = set()
        self.top_app = None
        self.new_apps = None

    def reset(self):
        self.turn_count = 0
        self.planner_history = []
        self.feedback_history = []
        self.action_history = []
        self.planning_agent.reset()
        self.reflection_agent.reset()
        self.rag_agent.reset()
        self.lifelong_learning_agent.reset()

    def flush_messages(self):
        for agent in [self.planning_agent, self.reflection_agent]:
            # After every max_trajectory_length trajectories, remove messages from the start except the system prompt
            if len(agent.messages) > 2*self.max_trajectory_length + 1:
                # Remove the user message and assistant message, both are 1 because the elements will move back after 1 pop
                agent.remove_message_at(1)
                agent.remove_message_at(1)

    def calculate_tokens(self, messages):

        num_input_images = 0
        output_message = messages[-1]

        input_message = messages[:-1]

        input_string = """"""
        for message in input_message:
            input_string += message["content"][0]["text"] + "\n"
            if len(message["content"]) > 1:
                num_input_images += 1

        input_text_tokens = osworld_utils.get_input_token_length(input_string)

        input_image_tokens = NUM_IMAGE_TOKEN*num_input_images

        output_tokens = osworld_utils.get_input_token_length(
            output_message["content"][0]["text"])

        return (input_text_tokens + input_image_tokens), output_tokens

    def process_input(self, obs):
        if self.observation_type == "a11y_tree":
            linearized_accessibility_tree = osworld_utils.linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                                       platform=self.platform)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = osworld_utils.trim_accessibility_tree(linearized_accessibility_tree,
                                                                                      self.a11y_tree_max_tokens)
            # Set current tree as the prev tree for the next iteration
            self.prev_atree = linearized_accessibility_tree

            return linearized_accessibility_tree

    def parse_actions(self, response: str, masks=None):
        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = osworld_utils.parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = osworld_utils.parse_code_from_string(response)
            elif self.action_space == 'fixed_space':
                actions = osworld_utils.parse_action_from_fixed_code(
                    response, self.linearized_accessiblity_tree)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def call_llm(self, agent):
        # Retry if fails
        max_retries = 3  # Set the maximum number of retries
        attempt = 0
        while attempt < max_retries:
            try:
                response = agent.get_response()
                break  # If successful, break out of the loop
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    print("Max retries reached. Handling failure.")
            time.sleep(1.)
        return response

    def retrieve_knowledge(self, instruction, current_state, engine):
        # Formulate query for searching
        try:
            query_path = os.path.join(working_dir, "kb", "formulate_query.json")
            formulate_query = json.load(open(query_path))
        except:
            formulate_query = {}

        if instruction in formulate_query.keys():
            search_query = formulate_query[instruction]
        else:
            self.rag_agent.add_system_prompt(self.rag_module_system_prompt.replace("TASK_DESCRIPTION", instruction).replace("ACCESSIBLITY_TREE", current_state))
            logger.info("RAG System Message: %s", self.rag_module_system_prompt.replace("TASK_DESCRIPTION", instruction).replace("ACCESSIBLITY_TREE", current_state))

            self.rag_agent.add_message("To use google search to get some useful information, first carefully analyze the accessibility tree of the current desktop UI state, then given the task instruction, formulate a question that can be used to search on the Internet for information in helping with the task execution.\nThe question should not be too general or too specific, but it should be based on the current desktop UI state (e.g., already open website or application). You should expect the google search will return you something useful based on the question. Since it is a desktop computer task, make sure to mention the corresponding task domain in the question and also mention the Ubuntu OS if you think the OS matters. Please ONLY provide the question.\nQuestion:")
            search_query = self.call_llm(self.rag_agent)
            self.rag_agent.add_message(search_query)
            search_query = search_query.strip().replace('"', '')

            formulate_query[instruction] = search_query
            with open(query_path, "w") as fout:
                json.dump(formulate_query, fout, indent=2)

        logger.info("SEARCH QUERY: %s", search_query)

        # Search from different engines
        if engine == 'llm':
            logger.info("Search Engine: LLM")
            file = os.path.join(working_dir, "kb", "llm_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info('Retrieved LLM Search Result: %s', exist_search_results[instruction])
                return search_query, exist_search_results[instruction]

            self.rag_agent.add_message(search_query)
            search_results = self.call_llm(self.rag_agent)
            self.rag_agent.add_message(search_results)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        elif engine == 'perplexica':
            logger.info("Search Engine: Perplexica Search")
            file = os.path.join(working_dir, "kb", "perplexica_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info('Retrieved Perplexica Search Result: %s', exist_search_results[instruction])
                return search_query, exist_search_results[instruction]

            search_results = query_to_perplexica(search_query)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        else:
            print("Search Engine Not Implemented!!!")

        logger.info("SEARCH RESULT: %s", search_results.strip())

        return search_query, search_results

    def generate_lifelong_learning_reflection(self, trajectory):

        # Create Reflection on whole trajectories for next round trial
        self.lifelong_learning_agent.add_system_prompt(self.lifelong_learning_system_prompt)
        self.lifelong_learning_agent.add_message(trajectory)
        lifelong_learning_reflection = self.call_llm(self.lifelong_learning_agent)

        return lifelong_learning_reflection

    def retrieve_lifelong_learning_reflection(self, instruction):

        try:
            lifelong_learning_reflection_dicts = json.load(open(os.path.join(working_dir, "kb", "lifelong_learning_knowledge_base.json")))
            lifelong_learning_reflection = lifelong_learning_reflection_dicts[instruction]
        except:
            lifelong_learning_reflection = "None"

        return lifelong_learning_reflection

    def retrieve_most_similar_knowledge(self, instruction):

        try:
            knowledge_base_dict = json.load(open(os.path.join(working_dir, "kb", "lifelong_learning_knowledge_base.json")))

            try:
                with open(os.path.join(working_dir, "kb", 'embeddings.pkl'), "rb") as f:
                    embeddings = pickle.load(f)
            except:
                embeddings = {}

            if instruction in embeddings.keys():
                instruction_embedding = embeddings[instruction]
            else:
                instruction_embedding = self.embedding_engine.get_embeddings(instruction)
                embeddings[instruction] = instruction_embedding

            candidate_embeddings = []
            for key in list(knowledge_base_dict.keys()):
                if key in embeddings.keys():
                    candidate_embedding = embeddings[key]
                else:
                    candidate_embedding = self.embedding_engine.get_embeddings(key)
                    embeddings[key] = candidate_embedding
                candidate_embeddings.append(candidate_embedding)
            candidate_embeddings = np.vstack(candidate_embeddings)

            with open(os.path.join(working_dir, "kb", 'embeddings.pkl'), "wb") as f:
                pickle.dump(embeddings, f)

            # instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            # candidate_embeddings = self.embedding_engine.get_embeddings(list(knowledge_base_dict.keys()))
            similarities = cosine_similarity(instruction_embedding, candidate_embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_instructions = [list(knowledge_base_dict.keys())[i] for i in sorted_indices]
            sorted_experiences = [list(knowledge_base_dict.values())[i] for i in sorted_indices]

            if sorted_instructions[0] != instruction:
                most_similar_task = sorted_instructions[0]
                retrieved_experience = sorted_experiences[0]
            else:
                most_similar_task = sorted_instructions[1]
                retrieved_experience = sorted_experiences[1]
        except:
            most_similar_task = "None"
            retrieved_experience = "None"

        return most_similar_task, retrieved_experience

    def knowledge_fusion(self, web_knowledge, most_similar_task, experience):
        self.rag_agent.add_message(f"**Web search result**:\n{web_knowledge}\nNote that the applications are already installed, so you do not need to install again, and the required files already exist.\n\n**Retrieved similar task experience**:\nSimilar task:{most_similar_task}\n{experience}\n\nBased on the web search result and the retrieved similar task experience, if you think the similar task experience is indeed useful to the main task, integrate it with the web search result. Provide the final knowledge in a numbered list.")
        integrated_knowledge  = self.call_llm(self.rag_agent)
        return integrated_knowledge

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        # Provide the top_app to the Grounding Agent to remove all other applications from the tree. At t=0, top_app is None
        agent = GroundingAgent()
        curr_atree = agent.linearize_and_annotate_tree(obs)
        self.active_apps = agent.get_current_applications(obs)

        # Get RAG knowledge, only update system message at t=0
        search_query = ''
        if self.turn_count == 0:
            search_query, retrieved_knowledge = self.retrieve_knowledge(instruction, current_state=curr_atree, engine=self.engine)
            most_similar_task, retrieved_experience = self.retrieve_most_similar_knowledge(search_query)
            logger.info("SIMILAR TASK EXPERIENCE: %s", most_similar_task + '\n' + retrieved_experience.strip())
            integrated_knowledge = self.knowledge_fusion(retrieved_knowledge, most_similar_task, retrieved_experience)
            logger.info("INTEGRATED KNOWLEDGE: %s", integrated_knowledge)
            instruction += f"\nYou may refer to some retrieved knowledge if you think they are useful.{integrated_knowledge}"

            self.planning_agent.add_system_prompt(
            self.planning_module_system_prompt
            .replace("TASK_DESCRIPTION", instruction))
            self.reflection_agent.add_system_prompt(
                self.reflection_module_system_prompt)

        # Clear older messages
        self.flush_messages()

        # Reflection generation
        reflection = None
        if self.enable_reflection and self.turn_count > 0:
            self.reflection_agent.add_message(
                'Task Description: ' + instruction + '\n' + 'Current Trajectory: ' + '\n\n'.join(self.planner_history) + '\n')
            reflection = self.call_llm(self.reflection_agent)
            self.reflections.append(reflection)
            self.reflection_agent.add_message(reflection)

            logger.info("REFLECTION: %s", reflection)

        # Plan Generation
        if reflection:
            self.planning_agent.add_message('\nYou may use the reflection on the previous trajectory: ' + reflection +
                                            f"\nAccessibility Tree: {curr_atree}\nnThe clipboard contains: {agent.clipboard}. Your notes: {agent.notes}. The current open applications are {agent.get_current_applications(obs)} and the active app is {agent.top_app}")
        else:
            self.planning_agent.add_message(
                f"Accessibility Tree: {curr_atree}\nThe clipboard contains: {agent.clipboard}. Your notes: {agent.notes}. The current open applications are {agent.get_current_applications(obs)} and the active app is {agent.top_app}")

        plan = self.call_llm(self.planning_agent)
        self.planner_history.append(plan)
        logger.info("PLAN: %s", plan)

        self.planning_agent.add_message(plan)

        # Calculate input and output tokens
        input_tokens, output_tokens = self.calculate_tokens(
            self.planning_agent.messages)

        # Set Cost based on GPT-4o
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        self.cost_this_turn += cost
        logger.info("COST: %s", self.cost_this_turn)

        # Extract code block from the plan
        plan_code = osworld_utils.parse_single_code_from_string(plan)
        exec_code = eval(plan_code)
        if agent.index_out_of_range_flag:
            plan_code = "agent.wait(1.0)"
            exec_code = eval(plan_code)
            agent.index_out_of_range_flag = False

        info = {
            'search_query': search_query,
            'plan': plan,
            'linearized_accessibility_tree': curr_atree,
            'plan_code': plan_code,
            'reflection': reflection,
            'num_input_tokens': input_tokens,
            'num_output_tokens': output_tokens,
            'cost': cost
        }
        self.turn_count+=1

        return info, [exec_code]
