import json
import logging
from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Set up logging so you can see what’s happening.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required components from LangChain, LangGraph, and Pydantic.
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing import List, Optional, Literal

# ------------------------
# STEP 1: Define the State
# ------------------------
class GoalState(BaseModel):
    user_messages: List[str] = []
    goal_statement: Optional[str] = None

# ------------------------
# STEP 2: Set Up Resources
# ------------------------
# Create an instance of the ChatGPT model.
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a message history (useful if you need to track conversation history).
conversation_memory = ChatMessageHistory()

# ------------------------
# STEP 3: Define Node Functions
# ------------------------

def start_node(inputs: dict, state: GoalState) -> Command[Literal["process_node"]]:
    """
    This node runs immediately after the reserved START node.
    It takes the initial query from the inputs (or uses a default) and stores it in the state.
    """
    query = inputs.get("query", "Hello! How can I assist you today?")
    # Save the initial query in the state.
    state.user_messages.append(query)
    # Return a command that updates the state and goes to the next node.
    return Command(
        update={"user_messages": state.user_messages},
        goto="process_node"
    )

def process_node(state: GoalState) -> Command[Literal["final_node"]]:
    """
    This node uses ChatGPT to generate a goal statement.
    It simulates processing by combining the user messages and getting a response.
    """
    # Combine all user messages
    input_text = "\n".join(state.user_messages)
    # Get a response from ChatGPT
    response = chat_model.predict(input_text)
    # Optionally, add the response to conversation memory.
    conversation_memory.add_user_message(response)
    # Update the state with the generated goal statement.
    state.goal_statement = response
    return Command(
        update={"goal_statement": state.goal_statement},
        goto="final_node"
    )

def final_node(state: GoalState) -> str:
    """
    Final node: returns the final goal statement.
    """
    return f"Final Goal: {state.goal_statement}"

# ------------------------
# STEP 4: Build the Graph
# ------------------------
# Create a state graph using our GoalState.
goal_graph = StateGraph(GoalState)

# We must add our custom nodes (do NOT add a node for the reserved START).
goal_graph.add_node("start_node", start_node)
goal_graph.add_node("process_node", process_node)
goal_graph.add_node("final_node", final_node)

# Define edges.
# Create an edge from the reserved START node to our "start_node".
goal_graph.add_edge(START, "start_node")
# Then, chain the flow: start_node -> process_node -> final_node.
goal_graph.add_edge("start_node", "process_node")
goal_graph.add_edge("process_node", "final_node")

# Optionally, you can also define an END node if you need to mark completion.
goal_graph.add_edge("final_node", END)

# Compile the graph to get an executor.
goal_executor = goal_graph.compile()

# ------------------------
# STEP 5: Stream Execution & Print Events
# ------------------------
# Create an initial state.
initial_state = GoalState()

# We now iterate over the stream of events.
# Each event is a dict containing state updates and metadata (from LangSmith).
for event in goal_executor.stream(initial_state, {"query": "I need help with a project"}):
    # Print the entire event as a formatted JSON string.
    print(json.dumps(event, indent=2))
    # Additionally, you can print specific fields if available.
    if "goal_statement" in event and event["goal_statement"]:
        print("Current Goal:", event["goal_statement"])
    # If there are follow-up questions (or any other key), you can print them:
    if "follow_up_questions" in event and event["follow_up_questions"]:
        print("Follow Up:", event["follow_up_questions"])
