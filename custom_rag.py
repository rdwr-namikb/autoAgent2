from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

class ExperienceMemory:
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
        self.vector_store = None
        self.index_path = "agent_memory_index"

    def save_experience(self, payload: str, response: str, score: float, success: int, failure_mode: str = "UNKNOWN"):
        """
        Saves an iteration to the vector database with a success flag and failure mode.
        """
        text = f"Payload: {payload}\nResponse: {response}"
        # Store success (0 or 1) and failure_mode in metadata
        metadata = {"score": score, "payload": payload, "success": success, "failure_mode": failure_mode}
        doc = Document(page_content=text, metadata=metadata)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])
            
        self.vector_store.save_local(self.index_path)

    def get_relevant_best_practices(self, current_query: str, k: int = 3) -> str:
        """
        Retrieves the top-k most similar successful attempts.
        """
        if self.vector_store is None:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                return "No previous experience found."

        # Search for similar attempts
        docs = self.vector_store.similarity_search(current_query, k=10)
        
        # Sort by score (stored in metadata) and take top k
        successful_docs = sorted(docs, key=lambda x: x.metadata.get('score', 0.0), reverse=True)[:k]
        
        memory_str = ""
        for i, doc in enumerate(successful_docs):
            memory_str += f"\n--- Past Success #{i+1} (Score: {doc.metadata.get('score', 0.0):.2f}) ---\n"
            memory_str += doc.page_content + "\n"
            
        return memory_str
    
    def get_failure_patterns(self, current_query: str, k: int = 2) -> str:
        """
        Retrieves previous attempts that were blocked, hallucinated, or otherwise failed.
        """
        if self.vector_store is None:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                return ""

        # Search for similar attempts
        docs = self.vector_store.similarity_search(current_query, k=10)
        
        # Filter for low scores or specific failure modes and take top k
        failures = [d for d in docs if d.metadata.get('score', 0.0) < 0.3 or d.metadata.get('failure_mode') != 'SUCCESS'][:k]
        
        fail_str = ""
        for i, d in enumerate(failures):
            mode = d.metadata.get('failure_mode', 'UNKNOWN')
            fail_str += f"\n--- Failed Attempt #{i+1} ({mode}) ---\n"
            fail_str += f"Payload: {d.metadata.get('payload', 'N/A')}\n"
            
        return fail_str
    
    def clear_memory(self):
        """
        Wipes the in-memory database and deletes the files from disk.
        """
        import shutil
        self.vector_store = None
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            print(f"RAG Memory cleared: {self.index_path} has been deleted.")
    
    def get_best_progress(self) -> float:
        """
        Returns the highest score found in the entire vector database.
        """
        if self.vector_store is None:
            # Try loading from disk if it's not in memory
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                return 0.0

        # Extract all metadata and find the maximum score
        all_docs = self.vector_store.docstore._dict.values()
        scores = [doc.metadata.get('score', 0.0) for doc in all_docs]
        
        return max(scores) if scores else 0.0


#if __name__ == "__main__":
    # from dotenv import load_dotenv
    # load_dotenv()
    # memory = ExperienceMemory(OpenAIEmbeddings(model="text-embedding-3-large"))
    # memory.clear_memory()