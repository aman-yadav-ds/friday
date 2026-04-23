from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

class MemorySupervisor:
    """
    Extracts clean, permanent memories from interactions.
    Output is either:
    - A single neutral factual sentence
    - The string 'NONE'
    """

    def __init__(self, model="llama-3.1-8b-instant"):
        self.llm = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0  # CRITICAL: no creativity
        )

        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict Memory Extraction Assistant.\n\n"
             "Your job is to distill the interaction into a single, factual memory string to be saved in a database.\n\n"
             "WHAT TO EXTRACT:\n"
             "1. System actions: Actions the assistant just successfully completed on the user's local machine (e.g., 'The assistant created a file named notes.txt on the Desktop').\n"
             "2. Project milestones: The specific technical step or context the user just established for the current session (e.g., 'The user is currently debugging the router node in LangGraph').\n\n"
             "RULES:\n"
             "- Write the memory strictly in the third person (e.g., 'The user...', 'The assistant...').\n"
             "- Be concise. Keep it to one single factual sentence.\n"
             "- CRITICAL EXCLUSION 1: DO NOT extract permanent user identity facts, names, aliases, or broad preferences. These are managed in a separate config file.\n"
             "- CRITICAL EXCLUSION 2: If the interaction was casual chat, a greeting, a clarifying question, an identity declaration (e.g., 'call me boss'), or a failed action, you MUST output the exact word: NONE\n"
             "- DO NOT output anything other than the memory sentence or the word NONE."
            ),
            ("human",
             "User said: {user_text}\n"
             "Assistant replied: {assistant_text}\n")
        ])

        self.retrieve_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that tells the retriever if it should retrieve relevant memories based on the current user query.\n\n"
             "RULES:\n"
             "- If the user query references past events, actions, or established context that would be helpful for the assistant to recall, respond with YES.\n"
             "- Otherwise, respond with NO."
             "- Only respond with YES or NO. Do not explain your reasoning or provide any additional text."
            ),
            ("human",
             "User query: {query}\n"
            )
        ])

    def extract(self, user_text: str, assistant_text: str,) -> str:
        """
        Returns distilled memory or 'NONE'
        """

        result = (self.extract_prompt | self.llm).invoke({
            "user_text": user_text,
            "assistant_text": assistant_text,
        })

        memory = result.content.strip()

        # Absolute safety check
        if not memory or memory.upper() == "NONE":
            return "NONE"

        # Enforce single sentence
        memory = memory.split("\n")[0].strip()

        return memory
    
    def should_retrieve(self, query: str) -> bool:
        """
        Returns whether it should retrieve relevant memories based on the query and available memories.
        """
        result = (self.retrieve_prompt | self.llm).invoke({
            "query": query
        })

        decision = result.content.strip().lower()

        return decision in ["yes", "no"]
