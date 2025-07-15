import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gui_agents.s2.core.module import BaseModule
from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.utils.common_utils import (
    call_llm_safe,
    load_embeddings,
    load_knowledge_base,
    save_embeddings,
)
from gui_agents.s2.utils.query_perplexica import query_to_perplexica


class KnowledgeBase(BaseModule):
    def __init__(
        self,
        embedding_engine,
        local_kb_path: str,
        platform: str,
        engine_params: Dict,
        save_knowledge: bool = True,
    ):
        super().__init__(engine_params, platform)

        self.local_kb_path = local_kb_path

        # initialize embedding engine
        self.embedding_engine = embedding_engine

        # Initialize paths for different memory types
        self.episodic_memory_path = os.path.join(
            self.local_kb_path, self.platform, "episodic_memory.json"
        )
        self.narrative_memory_path = os.path.join(
            self.local_kb_path, self.platform, "narrative_memory.json"
        )
        self.embeddings_path = os.path.join(
            self.local_kb_path, self.platform, "embeddings.pkl"
        )

        # Initialize trajectory tracking
        self.task_trajectory = ""
        self.current_subtask_trajectory = ""
        self.current_search_query = ""

        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT.replace(
            "CURRENT_OS", self.platform
        )

        # All three agents share a generic RAG prompt that asks the agent to provide information for UI automation in CURRENT_OS
        self.query_formulator = self._create_agent(self.rag_module_system_prompt)
        self.llm_search_agent = self._create_agent(self.rag_module_system_prompt)
        self.knowledge_fusion_agent = self._create_agent(self.rag_module_system_prompt)

        self.narrative_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.TASK_SUMMARIZATION_PROMPT
        )
        self.episode_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )

        self.save_knowledge = save_knowledge

    def retrieve_knowledge(
        self, instruction: str, search_query: str, search_engine: str = "llm"
    ) -> Tuple[str, str]:
        """Retrieve knowledge using search engine
        Args:
            instruction (str): task instruction
            observation (Dict): current observation
            search_engine (str): search engine to use"""

        # Use search engine to retrieve knowledge based on the formulated query
        search_results = self._search(instruction, search_query, search_engine)

        return search_query, search_results

    def formulate_query(self, instruction: str, observation: Dict) -> str:
        """Formulate search query based on instruction and current state"""
        query_path = os.path.join(
            self.local_kb_path, self.platform, "formulate_query.json"
        )
        try:
            with open(query_path, "r") as f:
                formulate_query = json.load(f)
        except:
            formulate_query = {}

        if instruction in formulate_query:
            return formulate_query[instruction]

        self.query_formulator.reset()

        self.query_formulator.add_message(
            f"The task is: {instruction}\n"
            "To use google search to get some useful information, first carefully analyze "
            "the screenshot of the current desktop UI state, then given the task "
            "instruction, formulate a question that can be used to search on the Internet "
            "for information in helping with the task execution.\n"
            "The question should not be too general or too specific. Please ONLY provide "
            "the question.\nQuestion:",
            image_content=(
                observation["screenshot"] if "screenshot" in observation else None
            ),
            role="user",
        )

        search_query = self.query_formulator.get_response().strip().replace('"', "")
        print("search query: ", search_query)
        formulate_query[instruction] = search_query
        with open(query_path, "w") as f:
            json.dump(formulate_query, f, indent=2)

        return search_query

    def _search(self, instruction: str, search_query: str, search_engine: str) -> str:
        """Execute search using specified engine"""

        # Default to perplexica rag knowledge to see if the query exists
        file = os.path.join(
            self.local_kb_path, self.platform, f"{search_engine}_rag_knowledge.json"
        )

        try:
            with open(file, "r") as f:
                exist_search_results = json.load(f)
        except:
            exist_search_results = {}

        if instruction in exist_search_results:
            return exist_search_results[instruction]
        if search_engine.lower() == "llm":
            self.llm_search_agent.reset()
            # Use LLM's internal knowledge like a search engine
            self.llm_search_agent.add_message(search_query, role="user")
            search_results = self.llm_search_agent.get_response()
        elif search_engine.lower() == "perplexica":
            # Use perplexica to search for the query
            search_results = query_to_perplexica(search_query)
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

        exist_search_results[instruction] = search_results.strip()
        with open(
            os.path.join(
                self.local_kb_path,
                self.platform,
                f"{search_engine}_rag_knowledge.json",
            ),
            "w",
        ) as f:
            json.dump(exist_search_results, f, indent=2)

        return search_results

    def retrieve_narrative_experience(self, instruction: str) -> Tuple[str, str]:
        """Retrieve narrative experience using embeddings"""

        knowledge_base = load_knowledge_base(self.narrative_memory_path)
        if not knowledge_base:
            return "None", "None"

        embeddings = load_embeddings(self.embeddings_path)

        # Get or create instruction embedding
        instruction_embedding = embeddings.get(instruction)

        if instruction_embedding is None:
            instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            embeddings[instruction] = instruction_embedding

        # Get or create embeddings for knowledge base entries
        candidate_embeddings = []
        for key in knowledge_base:
            candidate_embedding = embeddings.get(key)
            if candidate_embedding is None:
                candidate_embedding = self.embedding_engine.get_embeddings(key)
                embeddings[key] = candidate_embedding

            candidate_embeddings.append(candidate_embedding)

        save_embeddings(self.embeddings_path, embeddings)

        similarities = cosine_similarity(
            instruction_embedding, np.vstack(candidate_embeddings)
        )[0]
        sorted_indices = np.argsort(similarities)[::-1]

        keys = list(knowledge_base.keys())
        idx = 1 if keys[sorted_indices[0]] == instruction else 0
        return keys[sorted_indices[idx]], knowledge_base[keys[sorted_indices[idx]]]

    def retrieve_episodic_experience(self, instruction: str) -> Tuple[str, str]:
        """Retrieve similar task experience using embeddings"""

        knowledge_base = load_knowledge_base(self.episodic_memory_path)
        if not knowledge_base:
            return "None", "None"

        embeddings = load_embeddings(self.embeddings_path)

        # Get or create instruction embedding
        instruction_embedding = embeddings.get(instruction)

        if instruction_embedding is None:
            instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            embeddings[instruction] = instruction_embedding

        # Get or create embeddings for knowledge base entries
        candidate_embeddings = []
        for key in knowledge_base:
            candidate_embedding = embeddings.get(key)
            if candidate_embedding is None:
                candidate_embedding = self.embedding_engine.get_embeddings(key)
                embeddings[key] = candidate_embedding

            candidate_embeddings.append(candidate_embedding)

        save_embeddings(self.embeddings_path, embeddings)

        similarities = cosine_similarity(
            instruction_embedding, np.vstack(candidate_embeddings)
        )[0]
        sorted_indices = np.argsort(similarities)[::-1]

        keys = list(knowledge_base.keys())
        idx = 1 if keys[sorted_indices[0]] == instruction else 0
        return keys[sorted_indices[idx]], knowledge_base[keys[sorted_indices[idx]]]

    def knowledge_fusion(
        self,
        observation: Dict,
        instruction: str,
        web_knowledge: str,
        similar_task: str,
        experience: str,
    ) -> str:
        """Combine web knowledge with similar task experience"""

        self.knowledge_fusion_agent.reset()

        self.knowledge_fusion_agent.add_message(
            f"Task: {instruction}\n"
            f"**Web search result**:\n{web_knowledge}\n\n"
            f"**Retrieved similar task experience**:\n"
            f"Similar task:{similar_task}\n{experience}\n\n"
            f"Based on the web search result and the retrieved similar task experience, "
            f"if you think the similar task experience is indeed useful to the main task, "
            f"integrate it with the web search result. Provide the final knowledge in a numbered list.",
            image_content=(
                observation["screenshot"] if "screenshot" in observation else None
            ),
            role="user",
        )
        return self.knowledge_fusion_agent.get_response()

    def save_episodic_memory(self, subtask_key: str, subtask_traj: str) -> None:
        """Save episodic memory (subtask level knowledge).

        Args:
            subtask_key (str): Key identifying the subtask
            subtask_traj (str): Trajectory/experience of the subtask
        """
        if not self.save_knowledge:
            return

        try:
            kb = load_knowledge_base(self.episodic_memory_path)
        except:
            kb = {}

        if subtask_key not in kb:
            subtask_summarization = self.summarize_episode(subtask_traj)
            kb[subtask_key] = subtask_summarization

            os.makedirs(os.path.dirname(self.episodic_memory_path), exist_ok=True)
            with open(self.episodic_memory_path, "w") as fout:
                json.dump(kb, fout, indent=2)

        return kb.get(subtask_key)

    def save_narrative_memory(self, task_key: str, task_traj: str) -> None:
        """Save narrative memory (task level knowledge).

        Args:
            task_key (str): Key identifying the task
            task_traj (str): Full trajectory/experience of the task
        """
        if not self.save_knowledge:
            return

        try:
            kb = load_knowledge_base(self.narrative_memory_path)
        except:
            kb = {}

        if task_key not in kb:
            task_summarization = self.summarize_narrative(task_traj)
            kb[task_key] = task_summarization

            os.makedirs(os.path.dirname(self.narrative_memory_path), exist_ok=True)
            with open(self.narrative_memory_path, "w") as fout:
                json.dump(kb, fout, indent=2)

        return kb.get(task_key)

    def initialize_task_trajectory(self, instruction: str) -> None:
        """Initialize a new task trajectory.

        Args:
            instruction (str): The task instruction
        """
        self.task_trajectory = f"Task:\n{instruction}"
        self.current_search_query = ""
        self.current_subtask_trajectory = ""

    def update_task_trajectory(self, meta_data: Dict) -> None:
        """Update the task trajectory with new metadata.

        Args:
            meta_data (Dict): Metadata from the agent's prediction
        """
        if not self.current_search_query and "search_query" in meta_data:
            self.current_search_query = meta_data["search_query"]

        self.task_trajectory += (
            "\n\nReflection:\n"
            + str(meta_data["reflection"])
            + "\n\n----------------------\n\nPlan:\n"
            + meta_data["executor_plan"]
        )

    def handle_subtask_trajectory(self, meta_data: Dict) -> None:
        """Handle subtask trajectory updates based on subtask status.

        Args:
            meta_data (Dict): Metadata containing subtask information

        Returns:
            bool: Whether the subtask was completed
        """
        subtask_status = meta_data["subtask_status"]
        subtask = meta_data["subtask"]
        subtask_info = meta_data["subtask_info"]

        if subtask_status in ["Start", "Done"]:
            # If there's an existing subtask trajectory, finalize it
            if self.current_subtask_trajectory:
                self.current_subtask_trajectory += "\nSubtask Completed.\n"
                subtask_key = self.current_subtask_trajectory.split(
                    "\n----------------------\n\nPlan:\n"
                )[0]
                self.save_episodic_memory(subtask_key, self.current_subtask_trajectory)
                self.current_subtask_trajectory = ""
                return True

            # Start new subtask trajectory
            self.current_subtask_trajectory = (
                f"Task:\n{self.current_search_query}\n\n"
                f"Subtask: {subtask}\n"
                f"Subtask Instruction: {subtask_info}\n"
                f"----------------------\n\n"
                f'Plan:\n{meta_data["executor_plan"]}\n'
            )
            return False

        elif subtask_status == "In":
            # Continue current subtask trajectory
            self.current_subtask_trajectory += (
                f'\n----------------------\n\nPlan:\n{meta_data["executor_plan"]}\n'
            )
            return False

    def finalize_task(self) -> None:
        """Finalize the task by saving any remaining trajectories."""
        # Save any remaining subtask trajectory
        if self.current_subtask_trajectory:
            self.current_subtask_trajectory += "\nSubtask Completed.\n"
            subtask_key = self.current_subtask_trajectory.split(
                "\n----------------------\n\nPlan:\n"
            )[0]
            self.save_episodic_memory(subtask_key, self.current_subtask_trajectory)

        # Save the complete task trajectory
        if self.task_trajectory and self.current_search_query:
            self.save_narrative_memory(self.current_search_query, self.task_trajectory)

        # Reset trajectories
        self.task_trajectory = ""
        self.current_subtask_trajectory = ""
        self.current_search_query = ""

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
        task_summarization = call_llm_safe(self.narrative_summarization_agent)

        return task_summarization
