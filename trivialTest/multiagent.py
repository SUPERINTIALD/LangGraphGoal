import random
from typing import Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState

@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])

@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)",
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]

def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        from langgraph.types import Command
        return Command(
            # navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # Pass the agent’s full internal message history plus our tool message.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent

# Use ChatOpenAI with GPT‑4 instead of Anthropic.
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

# Instantiate the OpenAI agent with GPT‑4.
model = ChatOpenAI(model="gpt-4", temperature=0.7)

# Define travel advisor tools and ReAct agent.
travel_advisor_tools = [
    get_travel_recommendations,
    make_handoff_tool(agent_name="hotel_advisor"),
]
travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    prompt=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

def call_travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "human"]]:
    # Invoke the ReAct agent with the full message history.
    response = travel_advisor.invoke(state)
    return Command(update=response, goto="human")

# Define hotel advisor tools and ReAct agent.
hotel_advisor_tools = [
    get_hotel_recommendations,
    make_handoff_tool(agent_name="travel_advisor"),
]
hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

def call_hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "human"]]:
    response = hotel_advisor.invoke(state)
    return Command(update=response, goto="human")

def human_node(
    state: MessagesState, config
) -> Command[Literal["hotel_advisor", "travel_advisor", "human"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")

    # Determine the active agent from the trigger metadata.
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")
    active_agent = langgraph_triggers[0].split(":")[1]

    return Command(
        update={
            "messages": [{
                "role": "human",
                "content": user_input,
            }]
        },
        goto=active_agent,
    )

# Build the graph.
builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", call_travel_advisor)
builder.add_node("hotel_advisor", call_hotel_advisor)
builder.add_node("human", human_node)

# Start with the travel advisor.
builder.add_edge(START, "travel_advisor")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# (Optional) Display the graph visualization.
# from IPython.display import display, Image
# display(Image(graph.get_graph().draw_mermaid_png()))
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("Graph visualization saved as 'graph.png'.")
import uuid
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = [
    # 1st round: user states they want somewhere warm.
    {
        "messages": [
            {"role": "user", "content": "i wanna go somewhere warm in the caribbean"}
        ]
    },
    # 2nd round: using the resume Command to simulate human input via interrupt.
    Command(
        resume="could you recommend a nice hotel in one of the areas and tell me which area it is."
    ),
    # 3rd round: another resume command.
    Command(
        resume="i like the first one. could you recommend something to do near the hotel?"
    ),
]

for idx, user_input in enumerate(inputs):
    print()
    print(f"--- Conversation Turn {idx + 1} ---")
    print()
    print(f"User: {user_input}")
    print()
    for update in graph.stream(
        user_input,
        config=thread_config,
        stream_mode="updates",
    ):
        for node_id, value in update.items():
            if isinstance(value, dict) and value.get("messages", []):
                last_message = value["messages"][-1]
                # Retrieve role and content robustly.
                if isinstance(last_message, dict):
                    role = last_message.get("role")
                    content = last_message.get("content")
                else:
                    role = getattr(last_message, "role", None)
                    content = getattr(last_message, "content", None)
                # Only print if the message role is not "human"
                if role != "human":
                    print(f"{node_id}: {content}")
