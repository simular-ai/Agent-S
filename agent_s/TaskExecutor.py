from agent_s.ProceduralMemory import PROCEDURAL_MEMORY
from agent_s.osworld.GroundingAgent import GroundingAgent
from agent_s.MultimodalEngine import OpenAIEmbeddingEngine
import json
import numpy as np
import pickle
import platform
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from agent_s.MultimodalAgent import LMMAgent
from agent_s import osworld_utils
from agent_s.query_perplexica import query_to_perplexica
import re
import logging

from agent_s.osworld_utils import Node, calculate_tokens, call_llm_safe

logger = logging.getLogger("desktopenv.agent")
working_dir = os.path.dirname(os.path.abspath(__file__))


def sanitize_code(code):
    # This pattern captures the outermost double-quoted text
    if "\n" in code:
        pattern = r'(".*?")'
        # Find all matches in the text
        matches = re.findall(pattern, code, flags=re.DOTALL)
        if matches:
            # Replace the first occurrence only
            first_match = matches[0]
            code = code.replace(first_match, f'"""{first_match[1:-1]}"""', 1)
    return code


def extract_first_agent_function(code_string):
    # Regular expression pattern to match 'agent' functions with any arguments, including nested parentheses
    pattern = r'agent\.[a-zA-Z_]+\((?:[^()\'"]|\'[^\']*\'|"[^"]*")*\)'

    # Find all matches in the string
    matches = re.findall(pattern, code_string)

    # Return the first match if found, otherwise return None
    return matches[0] if matches else None


class Executor:
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: GroundingAgent,
        search_engine: str = "perplexica",
        enable_reflection: bool = True,
        use_subtask_experience: bool = True,
        experiment_type: str = "osworld",
    ):
        self.grounding_agent = grounding_agent

        self.enable_reflection = enable_reflection
        self.engine_params = engine_params
        self.search_engine = search_engine
        self.use_subtask_experience = use_subtask_experience
        self.experiment_type = experiment_type
        self.reset()

    def flush_messages(self, n):
        # After every max_trajectory_length trajectories, remove messages from the start except the system prompt
        for agent in [self.generator_agent]:
            if len(agent.messages) > 2 * n + 1:
                # Remove the user message and assistant message, both are 1 because the elements will move back after 1 pop
                agent.remove_message_at(1)
                agent.remove_message_at(1)

    def reset(self):
        self.generator_agent = LMMAgent(self.engine_params)
        self.reflection_agent = LMMAgent(self.engine_params)
        self.rag_agent = LMMAgent(self.engine_params)
        self.embedding_engine = OpenAIEmbeddingEngine()

        if self.experiment_type == "osworld":
            current_os = 'Ubuntu'
        elif self.experiment_type == "windowsagentarena":
            current_os = 'Windows 11'
        elif self.experiment_type == "openaci":
            if platform.system() == "Linux":
                current_os = 'Ubuntu'
            elif platform.system() == "Windows":
                current_os = 'Windows 11'
            elif platform.system() == "Darwin":
                current_os = 'MacOS'
        
        self.generator_system_prompt = PROCEDURAL_MEMORY.construct_procedural_memory(
            GroundingAgent
        ).replace("CURRENT_OS", current_os)
        self.reflection_module_system_prompt = (
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )
        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT.replace("CURRENT_OS", current_os)   
        
        self.turn_count = 0
        self.planner_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.tree_inputs = []
        self.screenshot_inputs = []

    def retrieve_similar_subtask_experience(self, instruction):

        try:
            knowledge_base_dict = json.load(
                open(
                    os.path.join(
                        working_dir, "kb", self.experiment_type, "subtask_experience_knowledge_base.json"
                    )
                )
            )

            try:
                with open(os.path.join(working_dir, "kb", "embeddings.pkl"), "rb") as f:
                    embeddings = pickle.load(f)
            except:
                embeddings = {}

            if instruction in embeddings.keys():
                instruction_embedding = embeddings[instruction]
            else:
                instruction_embedding = self.embedding_engine.get_embeddings(
                    instruction
                )
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

            with open(os.path.join(working_dir, "kb", self.experiment_type, "embeddings.pkl"), "wb") as f:
                pickle.dump(embeddings, f)

            similarities = cosine_similarity(
                instruction_embedding, candidate_embeddings
            )[0]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_instructions = [
                list(knowledge_base_dict.keys())[i] for i in sorted_indices
            ]
            sorted_experiences = [
                list(knowledge_base_dict.values())[i] for i in sorted_indices
            ]

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

    def retrieve_subtask_knowledge(self, instruction, current_state, engine):
        # Formulate query for searching
        try:
            query_path = os.path.join(working_dir, "kb", self.experiment_type, "formulate_query.json")
            formulate_query = json.load(open(query_path))
        except:
            formulate_query = {}

        if instruction in formulate_query.keys():
            search_query = formulate_query[instruction]
        else:
            self.rag_agent.add_system_prompt(
                self.rag_module_system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                ).replace("ACCESSIBLITY_TREE", current_state)
            )
            # logger.info("RAG System Message: %s", self.rag_module_system_prompt.replace(
            #     "TASK_DESCRIPTION", instruction).replace("ACCESSIBLITY_TREE", current_state))

            self.rag_agent.add_message(
                "To use google search to get some useful information, first carefully analyze the accessibility tree of the current desktop UI state, then given the task instruction, formulate a question that can be used to search on the Internet for information in helping with the task execution.\nThe question should not be too general or too specific, but it should be based on the current desktop UI state (e.g., already open website or application). You should expect the google search will return you something useful based on the question. Since it is a desktop computer task, make sure to mention the corresponding task domain in the question and also mention the Ubuntu OS if you think the OS matters. Please ONLY provide the question.\nQuestion:"
            )
            search_query = call_llm_safe(self.rag_agent)
            search_query = search_query.strip().replace('"', "")

            formulate_query[instruction] = search_query
            with open(query_path, "w") as fout:
                json.dump(formulate_query, fout, indent=2)

        logger.info("SUBTASK SEARCH QUERY: %s", search_query)

        # Search from different engines
        if engine == "llm":
            logger.info("Search Engine: LLM")
            file = os.path.join(working_dir, "kb", self.experiment_type, "llm_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info(
                    "Retrieved LLM Search Result: %s", exist_search_results[instruction]
                )
                return exist_search_results[instruction]

            self.rag_agent.add_message(search_query)
            search_results = call_llm_safe(self.rag_agent)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        elif engine == "perplexica":
            logger.info("Search Engine: Perplexica Search")
            file = os.path.join(working_dir, "kb", self.experiment_type, "perplexica_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info(
                    "Retrieved Perplexica Search Result: %s",
                    exist_search_results[instruction],
                )
                return exist_search_results[instruction]

            search_results = query_to_perplexica(search_query)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        else:
            print("Search Engine Not Implemented!!!")

        logger.info("SUBTASK SEARCH RESULT: %s", search_results.strip())

        return search_results

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

        self.active_apps = agent.get_current_applications(obs)

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
                    self.retrieve_similar_subtask_experience(subtask_query_key)
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
                self.generator_system_prompt.replace("SUBTASK_DESCRIPTION", subtask)
                .replace("TASK_DESCRIPTION", instruction)
                .replace("FUTURE_TASKS", ", ".join([f.name for f in future_tasks]))
                .replace("DONE_TASKS", ",".join(d.name for d in done_task))
            )

            self.reflection_agent.add_system_prompt(
                self.reflection_module_system_prompt
            )

        # Clear older messages - we keep full context. if you want to keep only the last n messages, you can use the flush_messages function
        # self.flush_messages(3) # flushes generator messages

        # Reflection generation
        reflection = None
        if self.enable_reflection and self.turn_count > 0:
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

        generator_message = (
            (
                f"\nYou may use the reflection on the previous trajectory: {reflection}\n"
                if reflection
                else ""
            )
            + f"Accessibility Tree: {tree_input}\n"
            f"Text Buffer = [{','.join(agent.notes)}]. "
            f"The current open applications are {agent.get_current_applications(obs)} and the active app is {agent.top_app}. "
        )

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
        plan_code = osworld_utils.parse_single_code_from_string(
            plan.split("Grounded Action")[-1]
        )
        plan_code = sanitize_code(plan_code)
        plan_code = extract_first_agent_function(plan_code)
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
