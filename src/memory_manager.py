from src.memory_store import MemoryStore

class MemoryManager:
    """
    Manages the storage and retrieval of conversation history/memories.
    Acts as the bridge between the LLM Engine and the Vector Database.
    """
    def __init__(self):
        self.store_db = MemoryStore()

    def store(self, user_msg: str, ai_msg: str):
        """
        Saves the interaction to the memory store.
        Format: "User: [msg] | AI: [msg]"
        """
        # We save the pair so the context is preserved.
        text_to_save = f"User: {user_msg}\nAI: {ai_msg}"
        try:
            self.store_db.add_memory(text_to_save)
            print(f"💾 Memory saved.")
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def retrieve(self, query: str) -> str:
        """
        Retrieves relevant memories based on the query.
        Returns a formatted string to be injected into the system prompt.
        """
        try:
            results = self.store_db.search_memory(query, n_results=1)
            if not results:
                return ""
            
            # Format results for the LLM
            formatted_memories = "\n".join([f"- {m}" for m in results])
            return formatted_memories
        except Exception as e:
            print(f"⚠️ Failed to retrieve memory: {e}")
            return ""

if __name__ == "__main__":
    # Test
    mm = MemoryManager()
    mm.retrieve("test")
