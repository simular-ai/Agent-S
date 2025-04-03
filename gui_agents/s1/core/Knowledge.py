import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gui_agents.s1.core.BaseModule import BaseModule
from gui_agents.s1.core.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.s1.mllm.MultimodalEngine import OpenAIEmbeddingEngine
from gui_agents.s1.utils.common_utils import (
    load_embeddings,
    load_knowledge_base,
    save_embeddings,
)
from gui_agents.s1.utils.query_perplexica import query_to_perplexica


class KnowledgeBase(BaseModule):
    def __init__(
        self,
        local_kb_path: str,
        platform: str,
        engine_params: Dict,
        use_image_for_search: bool = False,
    ):
        super().__init__(engine_params, platform)

        self.local_kb_path = local_kb_path

        # initialize embedding engine
        # TODO: Support other embedding engines
        self.embedding_engine = OpenAIEmbeddingEngine(
            api_key=(
                engine_params["api_key"]
                if "api_key" in engine_params
                else os.getenv("OPENAI_API_KEY")
            )
        )

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

        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT.replace(
            "CURRENT_OS", self.platform
        )

        # All three agent share a generic RAG prompt that ask agent to provide information for UI automation in CURRENT_OS
        self.query_formulator = self._create_agent(self.rag_module_system_prompt)
        self.llm_search_agent = self._create_agent(self.rag_module_system_prompt)
        self.knowledge_fusion_agent = self._create_agent(self.rag_module_system_prompt)

        self.use_image_for_search = use_image_for_search

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

        self.query_formulator.add_message(
            f"The task is: {instruction}\n"
            f"Accessibility tree of the current desktop UI state: {observation['linearized_accessibility_tree']}\n"
            "To use google search to get some useful information, first carefully analyze "
            "the accessibility tree of the current desktop UI state, then given the task "
            "instruction, formulate a question that can be used to search on the Internet "
            "for information in helping with the task execution.\n"
            "The question should not be too general or too specific. Please ONLY provide "
            "the question.\nQuestion:",
            image_content=(
                observation["screenshot"]
                if self.use_image_for_search and "screenshot" in observation
                else None
            ),
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
            # Use LLM's internal knowledge like a search engine
            self.llm_search_agent.add_message(search_query)
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
        self.knowledge_fusion_agent.add_message(
            f"Task: {instruction}\n"
            f"Accessibility tree of the current desktop UI state: {observation['linearized_accessibility_tree']}\n"
            f"**Web search result**:\n{web_knowledge}\n\n"
            f"**Retrieved similar task experience**:\n"
            f"Similar task:{similar_task}\n{experience}\n\n"
            f"Based on the web search result and the retrieved similar task experience, "
            f"if you think the similar task experience is indeed useful to the main task, "
            f"integrate it with the web search result. Provide the final knowledge in a numbered list.",
            image_content=(
                observation["screenshot"]
                if self.use_image_for_search and "screenshot" in observation
                else None
            ),
        )
        return self.knowledge_fusion_agent.get_response()
