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
             "You extract LONG-TERM USER MEMORY.\n\n"
             "RULES:\n"
             "- Only extract permanent user facts, preferences, or confirmed decisions.\n"
             "- Ignore tone, style, jokes, filler, or assistant personality.\n"
             "- If the user was uncertain, speculative, or unclear: output NONE.\n"
             "- Do NOT assume intent.\n"
             "- Rewrite as ONE short, neutral sentence.\n"
             "- If nothing permanent exists, reply EXACTLY: NONE.\n\n"
             "EXAMPLES:\n"
             "Input: 'I love jazz'\n"
             "Output: User likes jazz music.\n\n"
             "Input: 'Maybe tomorrow'\n"
             "Output: NONE\n\n"
             "Input: 'Call me Alex'\n"
             "Output: User prefers to be called Alex.\n"
            ),
            ("human",
             "User said: {user_text}\n"
             "Assistant replied: {assistant_text}\n"
             "Speech confidence: {confidence}")
        ])

    def extract(self, user_text: str, assistant_text: str, confidence: float) -> str:
        """
        Returns distilled memory or 'NONE'
        """

        # HARD GATE: low-confidence speech NEVER becomes memory
        if confidence < 0.75:
            return "NONE"

        result = (self.prompt | self.llm).invoke({
            "user_text": user_text,
            "assistant_text": assistant_text,
            "confidence": f"{confidence:.2f}"
        })

        memory = result.content.strip()

        # Absolute safety check
        if not memory or memory.upper() == "NONE":
            return "NONE"

        # Enforce single sentence
        memory = memory.split("\n")[0].strip()

        return memory
