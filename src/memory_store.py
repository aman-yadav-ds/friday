import chromadb
from chromadb.utils import embedding_functions
import os
import time
import uuid
from utils.helpers import read_yaml_config

class MemoryStore:
    def __init__(self, config_path="config/config.yaml"):
        self._config = read_yaml_config(config_path)
        self._memory_settings = self._config.get("memory_settings", {})
        self._path = self._memory_settings.get("path", "memory_db")
        self._n_results = self._memory_settings.get("n_results", 3)
        self._embedding_model = self._memory_settings.get("embedding_model", "all-MiniLM-L6-v2")
        # Initialize Client
        self.client = chromadb.PersistentClient(path=self._path)
        
        # Initialize Embedding Function (Local)
        # using all-MiniLM-L6-v2 which is small and effective for this
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self._embedding_model)
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name="emma_memory",
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

    def search_memory(self, query: str):
        """
        Searches for relevant memories.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=self._n_results
        )
        # Results is a dict with list of lists, we return the first list of documents
        if results["documents"]:
            return results["documents"][0]
        return []
