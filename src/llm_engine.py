import os
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from utils.tools.os_tools import check_folder, create_text_file


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
    print(f"Router is checking the Manager's last message...")
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        print(f"Router is routing to WORKER...")
        return "worker"
    else:
        print(f"Router is routing to END...")
        return "end"


# --- 2. The Brain ---
class Brain:
    def __init__(self):
        # self.llm = ChatOllama(
        #     model = "qwen2.5-coder:7b",
        #     temperature=0.5
        # )

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5
        )

        self.brain_with_tools = self.llm.bind_tools([check_folder, create_text_file])

        self.tools_by_name = {tool.name: tool for tool in [check_folder, create_text_file]}

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

        self.app = workflow.compile()

        print(f"Brain is ready to go!")



    def manager_node(self, state: AgentState):
        """
        This is the first stop on the board game.
        It looks at the history and decides: "Do I talk, or do I use a tool?"
        """
        print(f"Manager is Thinking...")

        self.history = state["messages"]

        response = self.brain_with_tools.invoke(self.history)

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
            # Get the name of the tool and the arguments (like the folder path)
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            #3. Find the actual tool function using the name
            selected_tool = self.tools_by_name.get(tool_name)

            try:
                # We use .invoke() to run LangChain tools 
                result = selected_tool.invoke(tool_args)
            except Exception as e:
                # If the tool breaks, we tell the Manager it broke instead of crashing
                result = f"Error running tool '{tool_name}': {str(e)}"
            
            # 4. Pack the result into a special ToolMessage
            # CRITICAL: We must include the tool_call_id so the Manager knows WHICH tool this answer belongs to.
            tool_mesage = ToolMessage(
                content=str(result),
                name = tool_name,
                tool_call_id = tool_call['id']
            )

            tool_result.append(tool_mesage)

        return {"messages": tool_result}
    
    def brain_is_braining(self, user_input: str):
        print(f"Starting Conversation...")
        # 1. We put the user's request into the backpack
        initial_backpack = {"messages": [HumanMessage(content=user_input)]}

        # 2. We start the app and watch it stream through the boxes
        final_backpack = self.app.invoke(initial_backpack)

        print("\nFinal Answer:")
        print(final_backpack["messages"][-1].content)


if __name__ == "__main__":
    brain = Brain()
    user_request = "Can you create a text file called 'secret.txt' with the word 'Agent' inside?"
    brain.brain_is_braining(user_request)

    

