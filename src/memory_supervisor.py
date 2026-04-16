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

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Memory Extraction Assistant.\n\n"
             "Your job is to distill the interaction into a single, factual memory string to be saved in a database.\n"
             "WHAT TO EXTRACT:\n"
             "1. Permanent user facts or preferences (e.g., 'User prefers Python', 'User lives in New York').\n"
             "2. Actions the assistant just successfully completed on the user's system (e.g., 'Assistant created a file named secret.txt on the Desktop').\n\n"
             "RULES:\n"
             "- Write the memory in the third person (e.g., 'The user...', 'The assistant...').\n"
             "- Be concise. Keep it to one factual sentence.\n"
             "- If the interaction was just casual chat or a failed action, output the exact word: NONE\n"
             "- DO NOT output anything other than the memory or the word NONE."
            ),
            ("human",
             "User said: {user_text}\n"
             "Assistant replied: {assistant_text}\n")
        ])

    def extract(self, user_text: str, assistant_text: str,) -> str:
        """
        Returns distilled memory or 'NONE'
        """

        result = (self.prompt | self.llm).invoke({
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
