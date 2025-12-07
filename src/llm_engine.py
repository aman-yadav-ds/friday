from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utils.tools.music_tool import SpotifyTool
import os

load_dotenv()

@tool
def play_music(query: str) -> str:
    """This tools takes query as the song name by author and plays it on spotify"""
    return SpotifyTool().play_music(query)

@tool
def stop_music() -> str:
    """This tool stops the music playing on spotify"""
    return SpotifyTool().stop_music()

class LLMEngine:
    """
    The 'Brain' of the operation. 🧠
    This is where the magic happens (and occasionally some hallucinations).
    Wraps the Gemini API using LangChain to give our agent some semblance of intelligence.
    """
    def __init__(self):
        # Initialize the Gemini client via LangChain.
        # We're using 'ChatGoogleGenerativeAI' because it plays nice with the free API key.
        # No credit card required, just pure, unadulterated AI power.
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest", 
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7 # A little creativity, but let's not get too crazy.
        )

        self.llm = self.llm.bind_tools([play_music, stop_music])

        # The "Personality" of the AI.
        # We tell it to be a helpful voice assistant, not a novelist.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a voice AI Agent named Friday. You are helpful, witty, and concise. "
                       "Your responses are spoken out loud, so keep them short and conversational. "
                       "Avoid markdown formatting like bold or bullet points unless absolutely necessary. "
                       "If asked about yourself, you can mention you are a brain in a jar powered by Gemini."),
            ("human", "{text}"),
        ])
        
        # Create the chain (Prompt -> LLM)
        self.chain = self.prompt | self.llm
        
    def generate_response_stream(self, text):
        """
        Sends your text to Gemini and returns a 'stream' of consciousness.
        Streaming is crucial here. It allows the 'Mouth' to start yapping while the 'Brain' 
        is still figuring out the end of the sentence. Makes it feel alive.
        """
        print("🧠 Thinking (with LangChain)...")
        # Stream the response so we can pipeline the audio generation.
        stream = self.chain.stream({"text": text})
        
        for chunk in stream:
            if chunk.type == "ai":
                if chunk.content:
                    yield chunk.content if hasattr(chunk, "content") else chunk
            

                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.args
                        
                        print(f"\n LLM requested tool: {tool_name}({tool_args})")

                        tool_obj: BaseTool = self.get_tool(tool_name)
                        tool_result = tool_obj.invoke(tool_args)

                        print(f"Tool result: {tool_result}")

                        follow_up = self.chain.invoke({"text": tool_result})

                        yield follow_up.content
            else:
                yield chunk.content if hasattr(chunk, "content") else chunk



    def get_tool(self, name:str):
        for tool in [play_music, stop_music]:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool {name} not found")

if __name__ == "__main__":
    llm = LLMEngine()
    for chunk in llm.generate_response_stream("Play WRONG by Chris Grey."):
        print(chunk)