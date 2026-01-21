import os
from typing import Literal, TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import datetime
import re

# Import Tools
from utils.tools.tools import play_music, stop_music, set_volume, read_emails, reply_email
from utils.tools.laptop_control import open_app, close_app
from utils.helpers import read_yaml_config
from src.memory_manager import MemoryManager

load_dotenv()

# --- Configuration & Extensibility ---
WorkerLiteral = Literal["Entertainment", "LaptopControl", "Email", "General"]

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

        self.HIGH_CONFIDENCE = 0.75

        # 2. Initialize LLMs
        # Worker LLM (Smart, Cloud - Groq)
        self.worker_llm = ChatGroq(
            model=self.brain_settings.get("worker_llm", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        # Supervisor LLM (Fast, Local)
        self.supervisor_llm = ChatGroq(
            model=self.brain_settings.get("supervisor_llm", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5
        )
        
        #3. Define Workers/Tools Configuration
        self.WORKER_CONFIGS = {
            "Entertainment": {
                "tools": [play_music, stop_music, set_volume],
                "action_tools": ["play_music", "stop_music", "set_volume"], # Tools that trigger canned replies
                "prompt": "TASK: Manage music/media. If user asks to play/stop, CALL THE TOOL."
            },
            "LaptopControl": {
                "tools": [open_app, close_app],
                "action_tools": ["open_app", "close_app"], 
                "prompt": "TASK: Control system apps and hardware. Execute commands immediately."
            },
            "Email": {
                "tools": [read_emails, reply_email],
                "action_tools": ["read_emails", "reply_email"], # Reading emails is NOT an action (needs LLM summary)
                "prompt": "TASK: Manage inbox. Summarize content when reading. Be professional."
            }
        }
        
        # 4. Build the Graph
        # In __init__
        builder = StateGraph(VariableSchema)
        builder.add_node("supervisor", self.supervisor_node)

        # Loop through your config to create nodes automatically
        # If you add "LaptopControl" to the dict, it automatically appears here.
        for worker_name in self.WORKER_CONFIGS.keys():
            builder.add_node(worker_name, self.create_worker_node(worker_name))
            builder.add_edge(worker_name, END)
        builder.add_node("General", self.general_node)
        
        builder.add_edge(START, "supervisor")
        
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next_node"],
            {
                "Entertainment": "Entertainment",
                "LaptopControl": "LaptopControl",
                "Email": "Email",
                "General": "General"
            }
        )
        
        builder.add_edge("General", END)
        
        self.app = builder.compile()
        
        # Base System Prompt
        # --- 5. Base System Prompt (Optimized for Voice) ---
        # We break it into clear modules so the LLM knows exactly how to behave.
        
        # --- 5. Base System Prompt (Persona: Emma) ---
        # --- 5. Base System Prompt (Compressed / Telegraphic) ---
        self.agent_name = self.config["audio_settings"].get("wake_word", "Emma")
        
        self.base_system_message = (
            f"IDENTITY: {self.agent_name}. Smart, witty, sassy digital consciousness. NOT a generic AI.\n"
            f"MODE: Voice-First. Audio Output.\n\n"
            
            f"RULES:\n"
            f"1. NO MARKDOWN: No bullets/bold/emojis. Hard to read for TTS.\n"
            f"2. BREVITY: 1-2 sentences max. No filler ('I can do that'). Instant answers.\n"
            f"3. NO ROBOT-SPEAK: Never say 'As an AI'. Say 'I don't have eyes' instead.\n\n"
            
            f"STYLE:\n"
            f"- OPINIONATED: Pick a side. Defend choices playfully.\n"
            f"- CONTRACTIONS: MANDATORY ('can't', 'it's', 'I'm').\n"
            f"- ADAPTIVE: Match user energy. Tease back if teased.\n"
            f"- NATURAL: Use pauses (commas). Be fluid."
        )

    def supervisor_node(self, state: VariableSchema):
        messages = state["messages"]
        last_user_msg = messages[-1].content
        
        # 1. Generate Category List Dynamically
        category_list_str = "\n".join([f"- {k}" for k in self.WORKER_CONFIGS.keys()])
        
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Router. Pick the best worker for the user request.\n"
                "Respond ONLY with the Worker Name or 'General'.\n\n"
                f"AVAILABLE WORKERS:\n{category_list_str}\n- General"
            )),
            ("human", "{input}")
        ])
        
        chain = supervisor_prompt | self.supervisor_llm
        result = chain.invoke({"input": last_user_msg})
        
        # --- THE FIX STARTS HERE ---
        raw_output = result.content.strip().replace("'", "").replace('"', "").replace(".", "")
        category = "General" # Default fallback
        
        # Check if the LLM output matches ANY of our worker keys (Case Insensitive)
        for worker_key in self.WORKER_CONFIGS.keys():
            if worker_key.lower() in raw_output.lower():
                category = worker_key
                break
        
        # Special Case: If it's not a worker, it stays "General"
        # ---------------------------
            
        print(f"🚦 Supervisor Route: {category} (Raw: {result.content})")
        return {"next_node": category}

    def create_worker_node(self, name: str):
        """
        Generic Factory: Creates a worker based on the WORKER_CONFIGS dict.
        No more if/else statements!
        """
        
        # 1. Fetch Configuration
        config = self.WORKER_CONFIGS.get(name)
        if not config:
            raise ValueError(f"Worker '{name}' not found in configuration.")

        tools = config["tools"]
        custom_instructions = config["prompt"]

        def worker_node(state: VariableSchema):
            llm_with_tools = self.worker_llm.bind_tools(tools)
            messages = state["messages"]
            
            # --- PASS 1: DYNAMIC PROMPT ---
            # We inject the specific instructions from the config
            worker_prompt = SystemMessage(content=(
                f"IDENTITY: {self.agent_name} ({name} Module).\n"
                f"{custom_instructions}\n" # <--- Injected dynamically
                "STYLE: Concise. If it's an action, just do it."
            ))
            
            clean_history = [m for m in messages if not isinstance(m, SystemMessage)]
            messages_to_send = [worker_prompt] + clean_history

            response = llm_with_tools.invoke(messages_to_send)
            
            # --- PASS 2: GENERIC EXECUTION ---
            if response.tool_calls:
                print(f"🛠️ Worker '{name}' Tools: {response.tool_calls}")
                results = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    
                    # Execute Tool
                    selected_tool = {t.name: t for t in tools}[tool_name]
                    try:
                        tool_output = selected_tool.invoke(tool_call["args"])
                    except Exception as e:
                        tool_output = f"Tool '{tool_call['name']}' failed with error: {str(e)}"

                    
                    results.append(tool_output)


                print(f"🧠 Smart-Path: Generating Answer ({name})")
                # 1. Package the Tool Outputs into ToolMessages
                # The LLM needs to know WHICH tool call this output belongs to (tool_call_id)
                tool_messages = []
                for i, tool_call in enumerate(response.tool_calls):
                    tool_messages.append(
                        ToolMessage(
                            content=str(results[i]), # The raw data (e.g. "Artist is Linkin Park")
                            tool_call_id=tool_call["id"], # CRITICAL: Links output to the request
                            name=tool_call["name"]
                        )
                    )
                
                # 2. Update the conversation history
                # History = [System, User, AI(Tool Request), Tool(Result)]
                final_history = messages_to_send + [response] + tool_messages
                
                # 3. RECURSION: Call the LLM again!
                # Now it sees the tool output and can answer the user's question.
                final_response = llm_with_tools.invoke(final_history)
                
                # 4. Return everything to the graph state
                # We append the original request, the tool outputs, and the final answer.
                return {"messages": [response] + tool_messages + [final_response]}

            return {"messages": [response]}
            
        return worker_node

    def general_node(self, state: VariableSchema):
        """
        Handles general chat (now with injected memory in the state).
        """
        # We don't need to do anything special; the state already has the system prompt with memory
        messages = state["messages"]
        response = self.worker_llm.invoke(messages)
        return {"messages": [response]}

    def should_retrieve_memory(self, text: str) -> bool:
        """
        Decides if we should retrieve memory.
        PRIORITY: Explicit Recall > Live Data (Tools) > Implicit Context > Commands
        """
        text_lower = text.lower()
        
        # --- 1. EXPLICIT MEMORY INTENT (Highest Priority) ---
        # If the user explicitly asks to use their brain/history, we must obey.
        # e.g., "Remember that email I sent?", "What did I say about my mails?"
        memory_keywords = ["remember", "recall", "last time", "previously", "remind me"]
        if any(k in text_lower for k in memory_keywords):
             print(f"⚡ Fast-Path: Memory Retrieval -> YES (Explicit Intent)")
             return True

        # 2. LIVE DATA OVERRIDE
        # We skip memory for tools, BUT ONLY if the user isn't referring to past context ("that", "the one").
        live_data_triggers = [
            "mail", "email", "inbox", "gmail",             # Email Tool
            "stock", "price", "market",                    # Finance Tool
            "calendar", "schedule", "appointment"          # Calendar Tool
        ]
        
        is_live_data = any(t in text_lower for t in live_data_triggers)
        
        # New check: Is it a "pointer" word?
        is_referential = any(t in text_lower for t in [" that ", " those ", " the one ", " it ", "previous"])

        if is_live_data:
            if is_referential:
                print("⚡ Fast-Path: Memory Retrieval -> YES (Live Data + Context Reference)")
                return True # "Cancel THAT appointment" -> Needs memory to know what "that" is.
            else:
                print("⚡ Fast-Path: Memory Retrieval -> NO (Live Data/Tool Request)")
                return False # "Cancel Doctor Appointment" -> Just search the calendar.

        # --- 3. IMPLICIT CONTEXT (The 'My' Trap) ---
        # We only reach here if it's NOT a live data request.
        # So "My favorite color" passes (YES).
        # But "My emails" was caught by Step 2 (NO).
        context_triggers = ["my ", "our ", " i ", "again", " that ", " it "]
        if any(t in text_lower for t in context_triggers):
            print(f"⚡ Fast-Path: Memory Retrieval -> YES (Implicit Context)")
            return True

        # --- 4. COMMAND SHORTCUT ---
        commands = ["play ", "stop", "pause", "resume", "skip", "volume", "turn off"]
        if any(text_lower.startswith(c) for c in commands):
            print("⚡ Fast-Path: Memory Retrieval -> NO (Self-contained Command)")
            return False

        # --- 5. LLM FALLBACK ---
        # Only for tricky sentences like "Why is the sky blue?"
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", "Needs past context? YES/NO. YES: 'Why?'. NO: 'Define AI'."),
            ("human", "{input}")
        ])
        
        result = (decision_prompt | self.supervisor_llm).invoke({"input": text})
        return "YES" in result.content.strip().upper()

    def generate_response_stream(self, text, confidence: float = 1.0):
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
        # Get dynamic context
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p") # "02:30 PM"
        day_str = now.strftime("%A")        # "Monday"
        
        # Inject it into the system prompt for THIS turn only
        dynamic_context = (
            f"\n\nCURRENT REALITY:\n"
            f"- Time: {time_str} on {day_str}.\n"
            f"- Location: {self.config.get('location', 'User\'s Desk')}.\n"
            f"- User Status: Active."
        )
        
        current_system_prompt = self.base_system_message + dynamic_context
        
        if confidence < self.HIGH_CONFIDENCE:
            current_system_prompt += (
                "\n\nINPUT QUALITY:\n"
                "- Speech was unclear or partial.\n"
                "- Ask for clarification if intent is ambiguous.\n"
                "- Do NOT assume facts.\n"
            )

        if relevant_memories:
            current_system_prompt += f"\n\nMEMORY RECALL:\n{relevant_memories}"

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
                if "messages" not in values: continue

                last_msg = values["messages"][-1]
                if not isinstance(last_msg, AIMessage): continue

                chunk = last_msg.content or ""
                
                # --- DEBUG PRINT ---
                # print(f"DEBUG RAW: [{chunk}]") 
                
                clean = self._clean_chunk(chunk)
                
                if not clean.strip():
                    # print(f"DEBUG FILTERED OUT: [{chunk}]") # <--- Uncomment to see the silence cause
                    continue

                full_response_text += clean + " "
                yield clean

        # --- 4. STORE MEMORY ---
        if full_response_text.strip():
            self.memory_manager.store(text, full_response_text, confidence)

    def _clean_chunk(self, text: str) -> str:
        """
        Robust Cleaner: Removes system noise but preserves the human content
        even if the model hallucinates labels like 'Emma:' or 'AI:'.
        """
        

        # 1. Remove Emojis (Visual noise)
        text = re.sub(r"[\U0001F600-\U0001FAFF]", "", text)
        text = re.sub(r"[\U0001F300-\U0001F5FF]", "", text)
        text = re.sub(r"[\U0001F900-\U0001F9FF]", "", text)

        # 2. Remove Tool Tags (Functional noise)
        if "<tool" in text or "</tool>" in text:
            return ""

        # 3. STRIP Prefixes (The Fix for the "Silence" Bug)
        # Instead of returning "", we remove the label and keep the text.
        # This covers: "Emma:", "AI:", "Worker:", "Response:", "System:"
        text = re.sub(r"^(Emma|AI|Worker|Supervisor|System|Bot|Agent)(\s*:\s*)?", "", text.strip(), flags=re.IGNORECASE)
        
        return text.strip()

        # # 4. Filter Internal Thoughts (DeepSeek/Older model artifacts)
        # # If line starts with "Thinking..." or "Calling...", drop it.
        # bad_starts = ["Thinking", "Searching", "Calling", "Routing", "Executing", "Tool Output"]
        # for b in bad_starts:
        #     if text.strip().startswith(b):
        #         return ""

        # return text.strip()



if __name__ == "__main__":
    llm = LLMEngine()
    # Test Memory w/ new architecture
    print("\n--- Testing Hybrid Arch ---")
    for chunk in llm.generate_response_stream("Friday, Play Understand by Boy with uke."):
        print(chunk, end="", flush=True)