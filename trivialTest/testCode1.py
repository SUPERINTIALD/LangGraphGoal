import json
from dotenv import load_dotenv
load_dotenv()

# Patch LangGraph's configuration so that interrupt() works outside a full runnable context.
import langgraph.config
langgraph.config.get_config = lambda: {"configurable": {}}

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing import List, Literal

# Define our state model.
class ConversationState(BaseModel):
    messages: List[str] = []

# A node that runs immediately after the reserved START node.
def start_override_node(inputs: dict, state: ConversationState) -> Command[Literal["human_node"]]:
    query = inputs.get("query", "No query provided")
    state.messages.append(f"Initial: {query}")
    return Command(update={"messages": state.messages}, goto="human_node")

# A human node that uses interrupt() to prompt the user for input.
def human_node(state: ConversationState) -> Command[Literal["process_node"]]:
    user_input = interrupt(value="Enter your message: ")
    state.messages.append(f"User: {user_input}")
    return Command(update={"messages": state.messages}, goto="process_node")

# A processing node that echoes back the last message.
def process_node(state: ConversationState) -> Command[Literal["final_node"]]:
    response = "Processed: " + state.messages[-1]
    state.messages.append(f"LLM: {response}")
    return Command(update={"messages": state.messages}, goto="final_node")

# A final node that returns the complete conversation as JSON.
def final_node(state: ConversationState) -> str:
    return json.dumps(state.messages, indent=2)

# Build the state graph.
graph = StateGraph(ConversationState)
graph.add_node("start_override_node", start_override_node)
graph.add_node("human_node", human_node)
graph.add_node("process_node", process_node)
graph.add_node("final_node", final_node)

# Create edges.
graph.add_edge(START, "start_override_node")
graph.add_edge("start_override_node", "human_node")
graph.add_edge("human_node", "process_node")
graph.add_edge("process_node", "final_node")
graph.add_edge("final_node", END)

# Compile the graph to get an executor.
executor = graph.compile()

# Try to patch the reserved START node if accessible.
if hasattr(executor, "nodes") and START in executor.nodes:
    executor.nodes[START].func = lambda inputs, state: Command(
        update={"messages": []},
        goto="start_override_node"
    )
else:
    # Fallback: set the entry point directly.
    executor.set_entry_point("start_override_node")

# Create an initial state.
state = ConversationState()

# Run the graph. When the human node is reached, interrupt() will prompt for input.
for event in executor.stream(state, {"query": "dummy"}):
    print(json.dumps(event, indent=2))
