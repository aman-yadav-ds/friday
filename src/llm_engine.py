import os
from typing import Literal, TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Import Tools
from utils.tools.entertainment import play_music, stop_music
from utils.helpers import read_yaml_config
from src.memory_manager import MemoryManager

load_dotenv()

# --- Configuration & Extensibility ---
WorkerLiteral = Literal["Entertainment", "General"]

class VariableSchema(TypedDict):
    """Schema for our graph state."""
    messages: Annotated[list[BaseMessage], "The conversation history"]
    next_node: str

class LLMEngine:
    """
    The 'Brain' of the operation. 🧠
    Powered by a Hybrid Architecture:
    - Supervisor: Local Llama-1B (Ollama)
    - Workers: Cloud Llama-70B (Groq)
    - Memory: External Manager (Chroma)
    """
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_yaml_config(config_path)
        self.brain_settings = self.config.get("brain_settings", {})

        # 1. Initialize Memory Manager
        self.memory_manager = MemoryManager()

        # 2. Initialize LLMs
        # Worker LLM (Smart, Cloud - Groq)
        self.worker_llm = ChatGroq(
            model=self.brain_settings.get("worker_llm", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        # Supervisor LLM (Fast, Local)
        self.supervisor_llm = ChatGroq(
            model=self.brain_settings.get("supervisor_llm", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5
        )
        
        # 3. Define Tools
        self.entertainment_tools = [play_music, stop_music]
        
        self.llm = self.worker_llm 
        
        # 4. Build the Graph
        builder = StateGraph(VariableSchema)
        
        builder.add_node("supervisor", self.supervisor_node)
        builder.add_node("Entertainment", self.create_worker_node("Entertainment", self.entertainment_tools))
        builder.add_node("General", self.general_node)
        
        builder.add_edge(START, "supervisor")
        
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next_node"],
            {
                "Entertainment": "Entertainment",
                "General": "General"
            }
        )
        
        builder.add_edge("Entertainment", END)
        builder.add_edge("General", END)
        
        self.app = builder.compile()
        
        # Base System Prompt
        self.base_system_message = (
            f"You are a voice AI Agent named {self.config["audio_settings"].get("wake_word", "Emma")}. You are helpful, witty, and concise. "
            "Your responses are spoken out loud, so keep them short and conversational."
        )

    def supervisor_node(self, state: VariableSchema):
        """
        Routes between Entertainment and General using Llama-3.2-1B.
        """
        messages = state["messages"]
        last_user_msg = messages[-1].content
        
        # --- LOCAL LLM ROUTING ---
        # Ask Llama-1B to classify
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Supervisor AI. Route the user's request. "
                "Respond ONLY with one of these words: Entertainment, General.\n"
                "Categories:\n"
                "- Entertainment: Playing music, stopping music, Spotify control, songs.\n"
                "- General: Everything else (chat, questions, memory, facts)."
            )),
            ("human", "{input}")
        ])
        
        chain = supervisor_prompt | self.supervisor_llm
        result = chain.invoke({"input": last_user_msg})
        
        # Clean the output
        category = result.content.strip().replace("'", "").replace('"', "").replace(".", "")
        
        # Fuzzy matching just in case the small model chats a bit
        if "entertainment" in category.lower(): 
            category = "Entertainment"
        else: 
            category = "General"
            
        print(f"🚦 Supervisor Route: {category} (Raw: {result.content})")
        return {"next_node": category}

    def create_worker_node(self, name: str, tools: list):
        """
        Factory function to create a worker node with tools.
        """
        def worker_node(state: VariableSchema):
            llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm
            
            messages = state["messages"]
            
            # --- CRITICAL FIX FOR ENTERTAINMENT ---
            # If we inject "I played X" from memory, the agent thinks it's done.
            # For Entertainment, we STRIP memory context and force fresh execution.
            if name == "Entertainment":
                # Filter out the SystemMessage with "Relevant Memories"
                clean_messages = [
                    m for m in messages 
                    if not (isinstance(m, SystemMessage) and "Relevant Memories" in str(m.content))
                ]
                # Prepend a focused DJ prompt
                dj_prompt = SystemMessage(content=(
                    f"{self.base_system_message}\n"
                    "You are the Entertainment Worker. "
                    "IGNORE past history. "
                    "If the user asks to play music, YOU MUST CALL the 'play_music' tool. "
                    "Do not just say you are playing it."
                ))
                messages_to_send = [dj_prompt] + [m for m in clean_messages if not isinstance(m, SystemMessage)]
            else:
                messages_to_send = messages

            response = llm_with_tools.invoke(messages_to_send)
            
            if response.tool_calls:
                print(f"🛠️ Worker '{name}' is calling tools: {response.tool_calls}")
                # Tool execution loop
                results = []
                for tool_call in response.tool_calls:
                    selected_tool = {t.name: t for t in tools}[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    print(f"   > Tool Output: {tool_output}")
                    results.append(tool_output)
                
                final_prompt = messages_to_send + [response, HumanMessage(content=str(results))]
                final_response = self.llm.invoke(final_prompt)
                return {"messages": [response, final_response]} 
            
            return {"messages": [response]}
            
        return worker_node

    def general_node(self, state: VariableSchema):
        """
        Handles general chat (now with injected memory in the state).
        """
        # We don't need to do anything special; the state already has the system prompt with memory
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def should_retrieve_memory(self, text: str) -> bool:
        """
        Decides if we should retrieve memory for this query.
        Returns True if yes, False if no.
        """
        # Heuristic: If it's short/action-oriented, likely NO.
        # But let's use the LLM for finer control as requested.
        
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Decision Engine. "
                "Should I retrieve past memories/context for this user query? "
                "Respond ONLY with 'YES' or 'NO'.\n"
                "Guidelines:\n"
                "- Music/Media Commands (e.g. 'Play X', 'Stop',) -> NO (Self-contained).\n"
                "- Questions/Chat/Facts (e.g. 'Try to retrieve/recall our chat', 'What did I say?', 'What did I tell you to do?', 'What are my preferences?', 'Remember What I like?') -> YES (Needs context).\n"
            )),
            ("human", "{input}")
        ])
        
        chain = decision_prompt | self.supervisor_llm
        result = chain.invoke({"input": text})
        decision = result.content.strip().upper()
        
        print(f"🤔 Memory Retrieval Needed? {decision}")
        return "YES" in decision

    def generate_response_stream(self, text):
        """
        Stream the human-speakable part of the assistant response.
        Filters out tool messages, logs, emojis, and system noise.
        """
        print("🧠 Thinking...")

        # --- 1. MEMORY RETRIEVAL ---
        relevant_memories = ""
        if self.should_retrieve_memory(text):
            relevant_memories = self.memory_manager.retrieve(text)
        else:
            print("🚫 Skipping Memory Retrieval (Action-based).")

        # --- 2. System prompt construction ---
        current_system_prompt = self.base_system_message
        if relevant_memories:
            current_system_prompt += f"\n\nRelevant Memories:\n{relevant_memories}"

        initial_state = {
            "messages": [
                SystemMessage(content=current_system_prompt),
                HumanMessage(content=text)
            ],
            "next_node": ""
        }

        full_response_text = ""

        # --- 3. STREAM RESPONSE ---
        for event in self.app.stream(initial_state):
            for node_name, values in event.items():
                if "messages" not in values:
                    continue

                last_msg = values["messages"][-1]
                if not isinstance(last_msg, AIMessage):
                    continue

                chunk = last_msg.content or ""
                clean = self._clean_chunk(chunk)
                if not clean.strip():
                    continue

                full_response_text += clean + " "
                yield clean

        # --- 4. STORE MEMORY ---
        if full_response_text.strip():
            self.memory_manager.store(text, full_response_text.strip())

    def _clean_chunk(self, text: str) -> str:
        """
        Remove emojis, tool-call messages, and system chatter.
        Leaves only human-speakable content.
        """
        import re

        # Remove emojis and unicode symbols (private & pictographic ranges)
        text = re.sub(r"[\U0001F600-\U0001FAFF]", "", text)
        text = re.sub(r"[\U0001F300-\U0001F5FF]", "", text)
        text = re.sub(r"[\U0001F900-\U0001F9FF]", "", text)

        # Remove tool-calls
        if text.strip().startswith("<tool") or text.strip().startswith("Tool Output"):
            return ""

        # Remove supervisor/debug chatter
        bad_prefixes = [
            "🧠", "🤔", "🚫", "🚦", "🛠️", 
            "Searching", "Tool Output", "Calling tool", 
            "Worker", "Supervisor"
        ]
        for b in bad_prefixes:
            if text.strip().startswith(b):
                return ""

        return text.strip()



if __name__ == "__main__":
    llm = LLMEngine()
    # Test Memory w/ new architecture
    print("\n--- Testing Hybrid Arch ---")
    for chunk in llm.generate_response_stream("Friday, Play Understand by Boy with uke."):
        print(chunk, end="", flush=True)