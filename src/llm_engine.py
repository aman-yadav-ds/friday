from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class LLMEngine:
    """
    The 'Brain' of the AI.
    Where the magic (and occasional hallucination) happens.
    Wraps the Gemini API using LangChain for extra brainpower.
    """
    def __init__(self):
        # Initialize the Gemini client via LangChain
        # We use the 'ChatGoogleGenerativeAI' class because it plays nice with the free API key.
        # No credit card required, just pure AI goodness.
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest", 
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )

        # The "Personality" of the AI.
        # We tell it to be a helpful voice assistant, not a novel writer.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful, witty, and concise voice assistant. "
                       "Your responses are spoken out loud, so keep them short and conversational. "
                       "Avoid markdown formatting like bold or bullet points unless absolutely necessary. "
                       "If asked about yourself, you can mention you are a brain in a jar powered by Gemini."),
            ("human", "{text}"),
        ])
        
        # Create the chain (Prompt -> LLM)
        self.chain = self.prompt | self.llm
        
    def generate_response_stream(self, text):
        """
        Sends your text to Gemini and returns a 'stream' of thoughts.
        Streaming is crucial. It lets us start speaking while the AI is still 
        figuring out the end of the sentence. Makes it feel alive.
        """
        print("🧠 Thinking (with LangChain)...")
        # We stream the response so the 'Mouth' can start yapping immediately.
        return self.chain.stream({"text": text})