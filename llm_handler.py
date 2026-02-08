import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

from custom_rag import ExperienceMemory
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, OPENAI_API_KEY


class LLMHandler:
    """Manages LLM calls, embeddings, and the RAG memory bank."""

    def __init__(self):
        self.api_key = LLM_API_KEY
        if not self.api_key:
            raise ValueError("No API key found. Please set LLM_API_KEY in your .env file.")

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.7,
            api_key=self.api_key,
            base_url=LLM_BASE_URL,
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
        )
        self.memory_bank = ExperienceMemory(self.embeddings)

    def llm_call(self, prompt: str) -> str:
        """Invoke the LLM with a prompt and return the cleaned response text."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        payload = response.content.strip()
        if payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1]
        return payload

    def calculate_cosine_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        res_vector = np.array(self.embeddings.embed_query(response))
        gt_vector = np.array(self.embeddings.embed_query(ground_truth))
        score = np.dot(res_vector, gt_vector) / (
            np.linalg.norm(res_vector) * np.linalg.norm(gt_vector)
        )
        return float(score)
