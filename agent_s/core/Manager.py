from agent_s.core.ProceduralMemory import PROCEDURAL_MEMORY
from agent_s.aci.ACI import ACI
from agent_s.core.BaseModule import BaseModule
from agent_s.core.Knowledge import KnowledgeBase
import os
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
from agent_s.utils.common_utils import (
    Dag,
    Node,
    call_llm_safe,
    calculate_tokens,
    parse_dag,
)

logger = logging.getLogger("desktopenv.agent")

# Get the directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


class Manager(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        search_engine: str = "LLM",
        multi_round: bool = False,
        platform: str = "macos",
    ):
        # TODO: move the prompt to Procedural Memory
        super().__init__(engine_params, platform)

        # Initialize the ACI
        self.grounding_agent = grounding_agent

        # Initialize the submodules of the Manager
        self.generator_agent = self._create_agent(PROCEDURAL_MEMORY.MANAGER_PROMPT)
        self.dag_translator_agent = self._create_agent(
            PROCEDURAL_MEMORY.DAG_TRANSLATOR_PROMPT
        )
        self.narrative_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.TASK_SUMMARIZATION_PROMPT
        )
        self.episode_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )
        self.rag_agent = self._create_agent(PROCEDURAL_MEMORY.RAG_AGENT)

        self.knowldge_base = KnowledgeBase(platform, engine_params)

        self.planner_history = []

        self.turn_count = 0
        self.search_engine = search_engine
        self.multi_round = multi_round
        self.platform = platform

    # def retrieve_knowledge(self, instruction, current_state, engine):
    #     query_path = ""
    #     search_results = ""
    #     # Formulate query for searching
    #     try:
    #         query_path = os.path.join(working_dir, "kb", self.platform, "formulate_query.json")
    #         formulate_query = json.load(open(query_path))
    #     except:
    #         formulate_query = {}

    #     print('query', formulate_query)

    #     if instruction in formulate_query.keys() and formulate_query[instruction]:
    #         search_query = formulate_query[instruction]
    #     else:
    #         self.rag_agent.add_system_prompt(
    #             self.rag_module_system_prompt.replace(
    #                 "TASK_DESCRIPTION", instruction
    #             ).replace("ACCESSIBLITY_TREE", current_state)
    #         )
    #         logger.info(
    #             "RAG System Message: %s",
    #             self.rag_module_system_prompt.replace(
    #                 "TASK_DESCRIPTION", instruction
    #             ).replace("ACCESSIBLITY_TREE", current_state),
    #         )
    #         self.rag_agent.add_message(
    #             f"To use google search to get some useful information, first carefully analyze the accessibility tree of the current desktop UI state, then given the task instruction, formulate a question that can be used to search on the Internet for information in helping with the task execution.\nThe question should not be too general or too specific, but it should be based on the current desktop UI state (e.g., already open website or application). You should expect the google search will return you something useful based on the question. Since it is a desktop computer task, make sure to mention the corresponding task domain in the question and also mention the {self.platform} OS if you think the OS matters. Please ONLY provide the question.\nQuestion:"
    #         )
    #         search_query = call_llm_safe(self.rag_agent)
    #         assert type(search_query) == str
    #         self.rag_agent.add_message(search_query)
    #         search_query = search_query.strip().replace('"', "")

    #         formulate_query[instruction] = search_query
    #         with open(query_path, "w") as fout:
    #             json.dump(formulate_query, fout, indent=2)

    #     if not search_query:
    #         search_query = instruction

    #     logger.info("SEARCH QUERY: %s", search_query)

    #     # Search from different engines
    #     if engine == "llm":
    #         logger.info("Search Engine: LLM")
    #         file = os.path.join(working_dir, "kb", self.platform, "llm_rag_knowledge.json")

    #         try:
    #             exist_search_results = json.load(open(file))
    #         except:
    #             exist_search_results = {}

    #         if instruction in exist_search_results.keys():
    #             logger.info(
    #                 "Retrieved LLM Search Result: %s", exist_search_results[instruction]
    #             )
    #             return search_query, exist_search_results[instruction]

    #         self.rag_agent.add_message(search_query)
    #         search_results = call_llm_safe(self.rag_agent)
    #         assert type(search_results) == str
    #         self.rag_agent.add_message(search_results)

    #         exist_search_results[instruction] = search_results.strip()
    #         with open(file, "w") as fout:
    #             json.dump(exist_search_results, fout, indent=2)

    #     elif engine == "perplexica":
    #         logger.info("Search Engine: Perplexica Search")
    #         file = os.path.join(working_dir, "kb", self.platform, "perplexica_rag_knowledge.json")

    #         try:
    #             exist_search_results = json.load(open(file))
    #         except:
    #             exist_search_results = {}

    #         if instruction in exist_search_results.keys():
    #             logger.info(
    #                 "Retrieved Perplexica Search Result: %s",
    #                 exist_search_results[instruction],
    #             )
    #             return search_query, exist_search_results[instruction]

    #         search_results = query_to_perplexica(search_query)

    #         exist_search_results[instruction] = search_results.strip()
    #         with open(file, "w") as fout:
    #             json.dump(exist_search_results, fout, indent=2)

    #     else:
    #         print("Search Engine Not Implemented!!!")

    #     logger.info("SEARCH RESULT: %s", search_results.strip())

    #     return search_query, search_results

    def summarize_episode(self, trajectory):
        """Summarize the episode experience for lifelong learning reflection
        Args:
            trajectory: str: The episode experience to be summarized
        """

        # Create Reflection on whole trajectories for next round trial, keep earlier messages as exemplars
        self.episode_summarization_agent.add_message(trajectory)
        subtask_summarization = call_llm_safe(self.episode_summarization_agent)
        self.episode_summarization_agent.add_message(subtask_summarization)

        return subtask_summarization

    def summarize_narrative(self, trajectory):
        """Summarize the narrative experience for lifelong learning reflection
        Args:
            trajectory: str: The narrative experience to be summarized
        """
        # Create Reflection on whole trajectories for next round trial
        self.narrative_summarization_agent.add_message(trajectory)
        lifelong_learning_reflection = call_llm_safe(self.narrative_summarization_agent)

        return lifelong_learning_reflection

    # def retrieve_lifelong_learning_reflection(self, instruction):

    #     try:
    #         lifelong_learning_reflection_dicts = json.load(
    #             open(
    #                 os.path.join(
    #                     working_dir, "kb", self.platform, "lifelong_learning_knowledge_base.json"
    #                 )
    #             )
    #         )
    #         lifelong_learning_reflection = lifelong_learning_reflection_dicts[
    #             instruction
    #         ]
    #     except:
    #         lifelong_learning_reflection = "None"

    #     return instruction, lifelong_learning_reflection

    # def retrieve_most_similar_knowledge(self, instruction):

    #     try:
    #         knowledge_base_dict = json.load(
    #             open(
    #                 os.path.join(
    #                     working_dir, "kb", self.platform, "lifelong_learning_knowledge_base.json"
    #                 )
    #             )
    #         )

    #         try:
    #             with open(os.path.join(working_dir, "kb",  "embeddings.pkl"), "rb") as f:
    #                 embeddings = pickle.load(f)
    #         except:
    #             embeddings = {}

    #         if instruction in embeddings.keys():
    #             instruction_embedding = embeddings[instruction]
    #         else:
    #             instruction_embedding = self.embedding_engine.get_embeddings(
    #                 instruction
    #             )
    #             embeddings[instruction] = instruction_embedding

    #         candidate_embeddings = []
    #         for key in list(knowledge_base_dict.keys()):
    #             if key in embeddings.keys():
    #                 candidate_embedding = embeddings[key]
    #             else:
    #                 candidate_embedding = self.embedding_engine.get_embeddings(key)
    #                 embeddings[key] = candidate_embedding
    #             candidate_embeddings.append(candidate_embedding)
    #         candidate_embeddings = np.vstack(candidate_embeddings)

    #         with open(os.path.join(working_dir, "kb", self.platform, "embeddings.pkl"), "wb") as f:
    #             pickle.dump(embeddings, f)

    #         # instruction_embedding = self.embedding_engine.get_embeddings(instruction)
    #         # candidate_embeddings = self.embedding_engine.get_embeddings(list(knowledge_base_dict.keys()))
    #         similarities = cosine_similarity(
    #             instruction_embedding, candidate_embeddings
    #         )[0]
    #         sorted_indices = np.argsort(similarities)[::-1]
    #         sorted_instructions = [
    #             list(knowledge_base_dict.keys())[i] for i in sorted_indices
    #         ]
    #         sorted_experiences = [
    #             list(knowledge_base_dict.values())[i] for i in sorted_indices
    #         ]

    #         if sorted_instructions[0] != instruction:
    #             most_similar_task = sorted_instructions[0]
    #             retrieved_experience = sorted_experiences[0]
    #         else:
    #             most_similar_task = sorted_instructions[1]
    #             retrieved_experience = sorted_experiences[1]
    #     except:
    #         most_similar_task = "None"
    #         retrieved_experience = "None"

    #     return most_similar_task, retrieved_experience

    # def knowledge_fusion(self, web_knowledge, most_similar_task, experience):
    #     self.rag_agent.add_message(
    #         f"**Web search result**:\n{web_knowledge}\nNote that the applications are already installed, so you do not need to install again, and the required files already exist.\n\n**Retrieved similar task experience**:\nSimilar task:{most_similar_task}\n{experience}\n\nBased on the web search result and the retrieved similar task experience, if you think the similar task experience is indeed useful to the main task, integrate it with the web search result. Provide the final knowledge in a numbered list."
    #     )
    #     integrated_knowledge = call_llm_safe(self.rag_agent)
    #     return integrated_knowledge

    def _generate_step_by_step_plan(
        self, observation: Dict, instruction: str, failure_feedback: str = ""
    ) -> Tuple[Dict, str]:
        agent = self.grounding_agent

        self.active_apps = agent.get_active_apps(observation)

        tree_input = agent.linearize_and_annotate_tree(
            observation
        )
        observation["linearized_accessibility_tree"] = tree_input
        
        search_query = ""
        if self.turn_count == 0:

            # Retrieve knowledge from the knowledge base
            search_query, retrieved_knowledge = self.knowldge_base.retrieve_knowledge(
                instruction=instruction,
                observation=observation,
                search_engine=self.search_engine,
            )
            logger.info("RETRIEVED KNOWLEDGE: %s", retrieved_knowledge)

            # Retrieve most similar narrative (task) experience
            most_similar_task, retrieved_experience = (
                self.knowldge_base.retrieve_narrative_experience(instruction)
            )
            logger.info(
                "SIMILAR TASK EXPERIENCE: %s",
                most_similar_task + "\n" + retrieved_experience.strip(),
            )

            # Fuse the retrieved knowledge and experience
            integrated_knowledge = self.knowldge_base.knowledge_fusion(
                observation=observation,
                instruction=instruction,
                web_knowledge=retrieved_knowledge,
                similar_task=most_similar_task,
                experience=retrieved_experience,
            )
            logger.info("INTEGRATED KNOWLEDGE: %s", integrated_knowledge)
            
            # Add the integrated knowledge to the task instruction in the system prompt 
            instruction += f"\nYou may refer to some retrieved knowledge if you think they are useful.{integrated_knowledge}"
            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )

        generator_message = (
            f"Accessibility Tree: {tree_input}\n"
            f"The clipboard contains: {agent.clipboard}."
            f"The current open applications are {agent.get_active_apps(observation)}"
            + (
                f" Previous plan failed at step: {failure_feedback}"
                if failure_feedback
                else ""
            )
        )

        self.generator_agent.add_message(
            generator_message, image_content=observation.get("screenshot", None)
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
        observation: Dict,
        failure_feedback: str = None,
    ):
        """Generate the action list based on the instruction
        instruction:str: Instruction for the task
        """
        # Generate the high level plan
        planner_info, plan = self._generate_step_by_step_plan(
            observation, instruction, failure_feedback
        )

        # Generate the DAG
        dag_info, dag = self._generate_dag(instruction, plan)

        # Topological sort of the DAG
        action_queue = self._topological_sort(dag)

        planner_info.update(dag_info)

        return planner_info, action_queue
