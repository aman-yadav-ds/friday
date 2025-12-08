import chromadb
from chromadb.utils import embedding_functions
import os
import time
import uuid

class MemoryStore:
    def __init__(self, path="memory_db"):
        self.path = path
        # Initialize Client
        self.client = chromadb.PersistentClient(path=self.path)
        
        # Initialize Embedding Function (Local)
        # using all-MiniLM-L6-v2 which is small and effective for this
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name="friday_memory",
            embedding_function=self.ef
        )

    def add_memory(self, text: str):
        """
        Adds a new memory to the database.
        """
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[text],
            metadatas=[{"timestamp": time.time()}],
            ids=[doc_id]
        )

    def search_memory(self, query: str, n_results: int = 3):
        """
        Searches for relevant memories.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Results is a dict with list of lists, we return the first list of documents
        if results["documents"]:
            return results["documents"][0]
        return []
