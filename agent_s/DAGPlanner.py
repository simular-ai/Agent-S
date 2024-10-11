from agent_s.ProceduralMemory import PROCEDURAL_MEMORY
from agent_s.agent_s.osworld.GroundingAgent import GroundingAgent
from agent_s.MultimodalEngine import OpenAIEmbeddingEngine
import numpy as np
import json
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from agent_s.MultimodalAgent import LMMAgent
import logging
from agent_s.query_perplexica import query_to_perplexica
from collections import defaultdict
from agent_s.osworld_utils import Dag, Node
from agent_s.osworld_utils import call_llm_safe, calculate_tokens, parse_dag

logger = logging.getLogger("desktopenv.agent")

# Get the directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
file_path = os.path.join(working_dir, "kb", "formulate_query.json")

NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


class Planner:
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: GroundingAgent,
        search_engine: str = "perplexica",
        use_plan_cache: bool = False,
        multi_round: bool = False,
    ):
        # TODO: move the prompt to Procedural Memory
        self.generator_agent = LMMAgent(
            engine_params, system_prompt=PROCEDURAL_MEMORY.DAG_PLANNER_BASE
        )
        self.dag_translator_agent = LMMAgent(
            engine_params, system_prompt=PROCEDURAL_MEMORY.DAG_TRANSLATOR_PROMPT
        )
        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT
        self.lifelong_learning_agent = LMMAgent(engine_params)
        self.lifelong_learning_system_prompt = (
            PROCEDURAL_MEMORY.LIFELONG_LEARNING_REFLECTION
        )
        self.subtask_summarization_agent = LMMAgent(engine_params)
        self.subtask_summarization_system_prompt = (
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )
        self.rag_agent = LMMAgent(engine_params)
        self.embedding_engine = OpenAIEmbeddingEngine()
        self.planner_history = []

        self.grounding_agent = grounding_agent
        self.turn_count = 0
        self.search_engine = search_engine
        self.use_plan_cache = use_plan_cache
        self.multi_round = multi_round

        self.plan_cache_path = os.path.join(working_dir, "kb", "graph_agent_plans.json")
        self.dag_cache_path = os.path.join(working_dir, "kb", "graph_agent_dags.json")
        self.search_query_cache_path = os.path.join(
            working_dir, "kb", "formulate_query.json"
        )

        # open cache files
        if os.path.exists(self.plan_cache_path):
            with open(self.plan_cache_path, "r") as f:
                self.plan_cache = json.load(f)
        else:
            self.plan_cache = {}

        if os.path.exists(self.dag_cache_path):
            with open(self.dag_cache_path, "r") as f:
                self.dag_cache = json.load(f)
        else:
            self.dag_cache = {}

        if os.path.exists(self.search_query_cache_path):
            with open(self.search_query_cache_path, "r") as f:
                self.search_query_cache = json.load(f)
        else:
            self.search_query_cache = {}

    def retrieve_knowledge(self, instruction, current_state, engine):
        query_path = ""
        search_results = ""
        # Formulate query for searching
        try:
            query_path = os.path.join(working_dir, "kb", "formulate_query.json")
            formulate_query = json.load(open(query_path))
        except:
            formulate_query = {}

        if instruction in formulate_query.keys() and formulate_query[instruction]:
            search_query = formulate_query[instruction]
        else:
            self.rag_agent.add_system_prompt(
                self.rag_module_system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                ).replace("ACCESSIBLITY_TREE", current_state)
            )
            logger.info(
                "RAG System Message: %s",
                self.rag_module_system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                ).replace("ACCESSIBLITY_TREE", current_state),
            )

            self.rag_agent.add_message(
                "To use google search to get some useful information, first carefully analyze the accessibility tree of the current desktop UI state, then given the task instruction, formulate a question that can be used to search on the Internet for information in helping with the task execution.\nThe question should not be too general or too specific, but it should be based on the current desktop UI state (e.g., already open website or application). You should expect the google search will return you something useful based on the question. Since it is a desktop computer task, make sure to mention the corresponding task domain in the question and also mention the Ubuntu OS if you think the OS matters. Please ONLY provide the question.\nQuestion:"
            )
            search_query = call_llm_safe(self.rag_agent)
            assert type(search_query) == str
            self.rag_agent.add_message(search_query)
            search_query = search_query.strip().replace('"', "")

            formulate_query[instruction] = search_query
            with open(query_path, "w") as fout:
                json.dump(formulate_query, fout, indent=2)

        if not search_query:
            search_query = instruction

        logger.info("SEARCH QUERY: %s", search_query)

        # Search from different engines
        if engine == "llm":
            logger.info("Search Engine: LLM")
            file = os.path.join(working_dir, "kb", "llm_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info(
                    "Retrieved LLM Search Result: %s", exist_search_results[instruction]
                )
                return search_query, exist_search_results[instruction]

            self.rag_agent.add_message(search_query)
            search_results = call_llm_safe(self.rag_agent)
            assert type(search_results) == str
            self.rag_agent.add_message(search_results)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        elif engine == "perplexica":
            logger.info("Search Engine: Perplexica Search")
            file = os.path.join(working_dir, "kb", "perplexica_rag_knowledge.json")

            try:
                exist_search_results = json.load(open(file))
            except:
                exist_search_results = {}

            if instruction in exist_search_results.keys():
                logger.info(
                    "Retrieved Perplexica Search Result: %s",
                    exist_search_results[instruction],
                )
                return search_query, exist_search_results[instruction]

            search_results = query_to_perplexica(search_query)

            exist_search_results[instruction] = search_results.strip()
            with open(file, "w") as fout:
                json.dump(exist_search_results, fout, indent=2)

        else:
            print("Search Engine Not Implemented!!!")

        logger.info("SEARCH RESULT: %s", search_results.strip())

        return search_query, search_results

    def generate_subtask_summarization(self, trajectory):

        # Create Reflection on whole trajectories for next round trial
        self.subtask_summarization_agent.add_system_prompt(
            self.subtask_summarization_system_prompt
        )
        self.subtask_summarization_agent.add_message(trajectory)
        subtask_summarization = call_llm_safe(self.subtask_summarization_agent)
        self.subtask_summarization_agent.add_message(subtask_summarization)

        return subtask_summarization

    def generate_lifelong_learning_reflection(self, trajectory):

        # Create Reflection on whole trajectories for next round trial
        self.lifelong_learning_agent.add_system_prompt(
            self.lifelong_learning_system_prompt
        )
        self.lifelong_learning_agent.add_message(trajectory)
        lifelong_learning_reflection = call_llm_safe(self.lifelong_learning_agent)

        return lifelong_learning_reflection

    def retrieve_lifelong_learning_reflection(self, instruction):

        try:
            lifelong_learning_reflection_dicts = json.load(
                open(
                    os.path.join(
                        working_dir, "kb", "lifelong_learning_knowledge_base.json"
                    )
                )
            )
            lifelong_learning_reflection = lifelong_learning_reflection_dicts[
                instruction
            ]
        except:
            lifelong_learning_reflection = "None"

        return instruction, lifelong_learning_reflection

    def retrieve_most_similar_knowledge(self, instruction):

        try:
            knowledge_base_dict = json.load(
                open(
                    os.path.join(
                        working_dir, "kb", "lifelong_learning_knowledge_base.json"
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

            with open(os.path.join(working_dir, "kb", "embeddings.pkl"), "wb") as f:
                pickle.dump(embeddings, f)

            # instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            # candidate_embeddings = self.embedding_engine.get_embeddings(list(knowledge_base_dict.keys()))
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

    def knowledge_fusion(self, web_knowledge, most_similar_task, experience):
        self.rag_agent.add_message(
            f"**Web search result**:\n{web_knowledge}\nNote that the applications are already installed, so you do not need to install again, and the required files already exist.\n\n**Retrieved similar task experience**:\nSimilar task:{most_similar_task}\n{experience}\n\nBased on the web search result and the retrieved similar task experience, if you think the similar task experience is indeed useful to the main task, integrate it with the web search result. Provide the final knowledge in a numbered list."
        )
        integrated_knowledge = call_llm_safe(self.rag_agent)
        return integrated_knowledge

    def _generate_step_by_step_plan(
        self, initial_observation: Dict, instruction: str, failure_feedback: str = ""
    ) -> Tuple[Dict, str]:
        agent = self.grounding_agent

        self.active_apps = agent.get_current_applications(initial_observation)

        # Get RAG knowledge, only update system message at t=0
        # TODO: at each step after the failed plan run a new rag search - rag search should include feedback from the failed plan
        # Include all apps during planning stage
        tree_input = agent.linearize_and_annotate_tree(
            initial_observation, show_all=False
        )
        search_query = ""
        if self.turn_count == 0:
            search_query, retrieved_knowledge = self.retrieve_knowledge(
                instruction, current_state=tree_input, engine=self.search_engine
            )
            if self.multi_round:
                most_similar_task, retrieved_experience = (
                    self.retrieve_lifelong_learning_reflection(search_query)
                )
            else:
                most_similar_task, retrieved_experience = (
                    self.retrieve_most_similar_knowledge(search_query)
                )
            logger.info(
                "SIMILAR TASK EXPERIENCE: %s",
                most_similar_task + "\n" + retrieved_experience.strip(),
            )
            integrated_knowledge = self.knowledge_fusion(
                retrieved_knowledge, most_similar_task, retrieved_experience
            )
            logger.info("INTEGRATED KNOWLEDGE: %s", integrated_knowledge)
            instruction += f"\nYou may refer to some retrieved knowledge if you think they are useful.{integrated_knowledge}"
            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )

        generator_message = (
            f"Accessibility Tree: {tree_input}\n"
            f"The clipboard contains: {agent.clipboard}."
            f"The current open applications are {agent.get_current_applications(initial_observation)}"
            + (
                f" Previous plan failed at step: {failure_feedback}"
                if failure_feedback
                else ""
            )
        )

        self.generator_agent.add_message(
            generator_message, image_content=initial_observation["screenshot"]
        )

        logger.info("GENERATING HIGH LEVEL PLAN")

        plan = call_llm_safe(self.generator_agent)

        if plan == "":
            raise Exception("Plan Generation Failed - Fix the Prompt")

        logger.info("HIGH LEVEL STEP BY STEP PLAN: %s", plan)

        self.generator_agent.add_message(plan)

        self.planner_history.append(plan)

        self.turn_count += 1

        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)

        # Set Cost based on GPT-4o
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        planner_info = {
            "search_query": search_query,
            "goal_plan": plan,
            "num_input_tokens_plan": input_tokens,
            "num_output_tokens_plan": output_tokens,
            "goal_plan_cost": cost,
        }

        assert type(plan) == str

        return planner_info, plan

    def _generate_dag(self, instruction: str, plan: str) -> Tuple[Dict, Dag]:
        # Add initial instruction and plan to the agent's message history
        self.dag_translator_agent.add_message(
            f"Instruction: {instruction}\nPlan: {plan}"
        )

        logger.info("GENERATING DAG")

        # Generate DAG
        dag_raw = call_llm_safe(self.dag_translator_agent)

        dag = parse_dag(dag_raw)

        logger.info("Generated DAG: %s", dag_raw)

        self.dag_translator_agent.add_message(dag_raw)

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
        initial_observation: Dict,
        replan: bool,
        failure_feedback: str = None,
    ):
        """Generate the action list based on the instruction
        instruction:str: Instruction for the task
        """
        # Generate the high level plan
        if self.use_plan_cache and instruction in self.plan_cache and not replan:
            plan = self.plan_cache[instruction]
            search_query = self.search_query_cache[instruction]
            planner_info = {
                "search_query": search_query,
                "goal_plan": plan,
                "retrived_from_cache": True,
            }
        else:
            planner_info, plan = self._generate_step_by_step_plan(
                initial_observation, instruction, failure_feedback
            )

            self.plan_cache[instruction] = plan

        if self.use_plan_cache and instruction in self.dag_cache and not replan:
            dag_json = self.dag_cache[instruction]
            dag = Dag.model_validate_json(dag_json)
            dag_info = {"dag": str(dag_json), "retrived_from_cache": True}
        else:
            # Generate the DAG
            dag_info, dag = self._generate_dag(instruction, plan)

            dag_json = dag.model_dump_json()
            self.dag_cache[instruction] = dag_json

        # Save caches to disk
        with open(self.plan_cache_path, "w") as f:
            json.dump(self.plan_cache, f)

        with open(self.dag_cache_path, "w") as f:
            json.dump(self.dag_cache, f)

        # Topological sort of the DAG
        action_queue = self._topological_sort(dag)

        planner_info.update(dag_info)

        return planner_info, action_queue
