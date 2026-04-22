import asyncio
import os
import platform
from dotenv import load_dotenv

from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


from src.memory_manager import MemoryManager
from utils.helpers import read_yaml_config
from utils.tools.os_tools import check_folder, create_file


load_dotenv()  # Load environment variables from .env file

# --- 1. The Backpack ---
class AgentState(TypedDict):
    # 'add_messages' is the magic word here. 
    # It tells LangGraph: "When new messages arrive, APPEND them to the list, don't overwrite it."
    messages : Annotated[list[BaseMessage], add_messages]

def router(state: AgentState):
    """
    Looks at the Manager's last message.
    If it asked for a tool, route to 'worker'.
    If it just talked, route to END.
    """
    print(f"\nRouter is checking the Manager's last message...")
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        print(f"Router is routing to WORKER...")
        return "worker"
    else:
        print(f"Router is routing to END...")
        return "end"


# --- 2. The Brain ---
class Brain:
    def __init__(self, config_path="config/brain_config.yaml"):
        # --- Reading Config ---
        self._config = read_yaml_config(config_path)
        print(self._config)
        if not self._config:
            raise ValueError(f"Failed to load brain config from '{config_path}'")


        # --- Thinking Architecture ---
        # self.llm = ChatOllama(
        #     model = "qwen2.5-coder:7b",
        #     temperature=0.5
        # )

        self.llm = ChatGroq(
            model=self._config["model"].get("name", "llama-3.3-70b-versatile"),
            temperature=self._config["model"].get("temperature", 0.7)
        )

        # --- Memory ---
        self.memory_manager = MemoryManager()

        self.checkpointer = MemorySaver()

        self.brain_with_tools = self.llm.bind_tools([check_folder, create_file])

        self.tools_by_name = {tool.name: tool for tool in [check_folder, create_file]}

        # --- The Board Game ---
        workflow = StateGraph(AgentState)
        workflow.add_node("manager", self.manager_node)
        workflow.add_node("worker", self.worker_node)

        #Draw the arrows
        workflow.add_edge(START, "manager")

        # The Manager uses the router to decide where to go next
        workflow.add_conditional_edges("manager", 
            router,
            {
                "worker": "worker",
                "end": END
            }
        )

        # The Worker ALWAYS goes back to the Manager (The Loop!)
        workflow.add_edge("worker", "manager")

        self.app = workflow.compile(checkpointer=self.checkpointer)

        print(f"Brain is ready to go!")



    def manager_node(self, state: AgentState):
        """
        This is the first stop on the board game.
        It looks at the history and decides: "Do I talk, or do I use a tool?"
        """

        print(f"Manager is Thinking...")
        # 1. Get the Platform and Home Directory for System Prompt (This is CRITICAL info for the Manager to have when deciding how to use tools safely!)
        current_os = platform.system()
        home_directory = os.path.expanduser("~")

        # 2. Extract the user's actual prompt to search memory
        # We loop backward through the backpack to find the last thing the human said
        last_user_message = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        
        relevant_memories = ""

        # 3. Retrieve relevant memories from Chroma
        if last_user_message:
            # You can also use your should_retrieve_memory logic here if you want to save processing time!
            print(f"Manager is retrieving relevant memories for the user's last message: '{last_user_message}'")
            relevant_memories = self.memory_manager.retrieve(last_user_message)

        memory_context = f"\n\nPast Memories:\n{relevant_memories}\n\n" if relevant_memories else ""

        # 4. Build the System Prompt with the injected memories
        system_prompt = SystemMessage(content=(
            f"You are {self._config.get('name', 'Edith')}, a helpful and precise AI assistant.\n"
            f"You are running on a {current_os} operating system.\n"
            f"The user's true home directory is: {home_directory}\n"
            f"Recent memories that might be relevant to this conversation: {memory_context}\n"
            f"User Preferences: {self._config.get('user_preferences', {})}\n"
            f"CRITICAL RULES:\n"
            f"1. When using tools to access the file system, ALWAYS use absolute paths based on the home directory.\n"
            f"2. Never guess usernames like 'username' or 'user'.\n"
            f"3. DO NOT use the `create_file` tool to save user preferences, names, rules, or conversational memories. "
            f"Your memory is managed automatically by the system. Just acknowledge the user's request naturally in text."
        ))

        history = state["messages"]
        trimmed_history = history[-10:]  # Keep only the last 10 messages for context to save tokens
        messages_to_send = [system_prompt] + trimmed_history

        response = self.brain_with_tools.invoke(messages_to_send)

        return {"messages": response}

    def worker_node(self, state: AgentState):
        """
        This box ONLY runs if the Manager asked for a tool.
        It acts as the "Hands" to do the physical work.
        """
        print(f"Worker is running tools...")

        #1. Get the last message, which should be a ToolMessage
        last_message = state["messages"][-1]

        # We will store our tool results here
        tool_result = []

        #2. Loop through every tool the Manager asked for
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # --- ADD THIS DEBUG PRINT ---
            print(f"   [DEBUG] Manager called: {tool_name}")
            print(f"   [DEBUG] Arguments: {tool_args}")

            selected_tool = self.tools_by_name.get(tool_name)

            try:
                result = selected_tool.invoke(tool_args)
            except Exception as e:
                result = f"Error running tool '{tool_name}': {str(e)}"
            
            # --- ADD THIS DEBUG PRINT ---
            print(f"   [DEBUG] Tool returned: {str(result)[:150]}...") # Only print first 150 chars so it doesn't flood terminal
            
            tool_message = ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )

            tool_result.append(tool_message)

        return {"messages": tool_result}
    
    async def brain_is_braining(self, user_input: str, thread_id: str = "boss_thread"):
        """
        Streams the AI response for the 'Mouth' while
        accumulating the final answer to store in memory at the end.
        """
        print(f"Starting Conversation...")
        # 1. We put the user's request into the backpack
        initial_backpack = {"messages": [HumanMessage(content=user_input)]}
        # This config tell the checkpointer which memory to load
        config = {"configurable": {"thread_id": thread_id}}
        full_response_content = ""

        # Use atream to get node updates
        async for msg, metadata in self.app.astream(
            initial_backpack,
            config=config,
            stream_mode="messages"
        ):
            # We only care about messages coming from the 'manager' node
            # and we want to ignore tool calls (those shouldn't be spoken)
            if metadata["langgraph_node"] == "manager" and not isinstance(msg, ToolMessage):
                content = msg.content
                if content:
                    full_response_content += content
                    yield content  # Stream the Manager's thoughts to the Mouth as they come in

        # After the stream is done, store the full response.
        if full_response_content.strip():
            print(f"Full response from Manager: {full_response_content}")
            print(f"Storing the user's request and the final answer in memory...")
            self.memory_manager.store(user_input, full_response_content)

if __name__ == "__main__":
    async def test_memory():
        brain = Brain()
        
        # Turn 1
        print("\n--- Turn 1 ---")
        async for chunk in brain.brain_is_braining("Check the document folder and tell me if you see a txt file there.", thread_id="test_1"):
            print(chunk, end="", flush=True)

        # Turn 2 (Testing if she remembers Turn 1)
        print("\n--- Turn 2 ---")
        async for chunk in brain.brain_is_braining("Is there a txt file there?", thread_id="test_1"):
            print(chunk, end="", flush=True)

    asyncio.run(test_memory())