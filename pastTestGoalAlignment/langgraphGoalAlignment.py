import random
import uuid
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

#####################
#   Example Tools   #
#####################

@tool
def dummy_tool():
    """A dummy tool that just returns a random comment."""
    return random.choice(["Interesting idea!", "Needs more detail!", "Could be improved.", "This goal is valid"])

#####################
#   Agents (LLMs)   #
#####################

goal_creator_model = ChatOpenAI(model="gpt-4", temperature=0.7)
goal_validator_model = ChatOpenAI(model="gpt-4", temperature=0.0)

goal_creator_tools = [dummy_tool]
goal_creator_agent = create_react_agent(
    goal_creator_model,
    goal_creator_tools,
    prompt=(
        "You are a 'goal creation advisor'. "
        "You help the user refine their goal statement. "
        "If you need extra details or clarifications, call the 'dummy_tool' or ask the user. "
        "Always provide a short, clear statement of the refined goal in your final message."
    ),
)

goal_validator_tools = [dummy_tool]
goal_validator_agent = create_react_agent(
    goal_validator_model,
    goal_validator_tools,
    prompt=(
        "You are a 'goal validator'. "
        "You receive a proposed goal and check if it is feasible, unambiguous, and clearly stated. "
        "If changes are needed, ask the user or call the 'dummy_tool' for hints. "
        "Otherwise, confirm the goal is valid."
        "Check if the proposed goal is feasible, unambiguous, and clearly stated. "
        "If it is, include the word 'valid' in your final message; otherwise, ask for more details."

    ),
)

#####################
#   Node Functions  #
#####################

def call_goal_creator_advisor(
    state: MessagesState,
) -> Command[Literal["human"]]:
    """Calls the 'goal_creator_agent' to refine or create the user's goal."""
    response = goal_creator_agent.invoke(state)
    return Command(update=response, goto="human")

def call_goal_validator(
    state: MessagesState,
) -> Command[Literal["goal_satisfied"]]:
    """Calls the 'goal_validator_agent' to check the goal's feasibility and clarity."""
    response = goal_validator_agent.invoke(state)
    return Command(update=response, goto="goal_satisfied")

def goal_satisfied_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "human", END]]:
    """
    Checks if both LLM and user are satisfied. 
    If not, routes back to creation; if user says yes, goes to END.
    """
    # 1) Check if LLM is satisfied (looking for 'valid' in last AI message).
    llm_satisfied = False
    if state.messages:
        last_ai_msg = [m for m in state.messages if m["role"] == "ai"]
        if last_ai_msg and "valid" in last_ai_msg[-1]["content"].lower():
            llm_satisfied = True

    # 2) If LLM not satisfied, route back to the goal creator.
    if not llm_satisfied:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "The LLM is NOT satisfied with the goal yet. Let's refine more.",
            }]},
            goto="goal_creator_advisor"
        )

    # 3) LLM is satisfied, now ask the user.
    user_input = interrupt(value="LLM is satisfied. Are YOU satisfied? (yes/no):")
    if user_input.strip().lower() in ["yes", "y"]:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "Great! Goal is finalized.",
            }]},
            goto=END
        )
    else:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "Understood. Let's refine further.",
            }]},
            goto="goal_creator_advisor"
        )

def human_node(
    state: MessagesState, config
) -> Command[Literal["goal_creator_advisor", "goal_validator", "goal_satisfied", "human"]]:
    """
    A node for collecting user input.
    We read the active agent from the trigger metadata.
    If no trigger is provided, we default to 'goal_creator_advisor'.
    """
    user_input = interrupt(value="Ready for user input:")
    triggers = config["metadata"].get("langgraph_triggers", [])
    if len(triggers) != 1:
        active_agent = "goal_creator_advisor"
    else:
        active_agent = triggers[0].split(":")[1]
    return Command(
        update={"messages": [{"role": "human", "content": user_input}]},
        goto=active_agent,
    )

#####################
#   Build the Graph #
#####################

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("goal_creator_advisor", call_goal_creator_advisor)
builder.add_node("goal_validator", call_goal_validator)
builder.add_node("goal_satisfied", goal_satisfied_node)
builder.add_node("human", human_node)

# Edges
builder.add_edge(START, "goal_creator_advisor")
builder.add_edge("goal_creator_advisor", "human")
builder.add_edge("goal_validator", "goal_satisfied")
builder.add_edge("goal_satisfied", "goal_creator_advisor")
builder.add_edge("goal_satisfied", END)

# Compile the graph.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# PATCH the built-in __start__ node so it writes an initial message to `messages`.
if START in graph.nodes:
    def start_func(inputs: dict, state: MessagesState):
        # If 'messages' provided in inputs, use them; otherwise, use default.
        initial_messages = inputs.get("messages", [])
        if not initial_messages:
            initial_messages = [{
                "role": "user",
                "content": "Hello from __start__: Help me create a goal."
            }]
        return Command(update={"messages": initial_messages}, goto="goal_creator_advisor")
    graph.nodes[START].func = start_func

# Save a PNG of the graph visualization.
png_data = graph.get_graph().draw_mermaid_png()
with open("graphGoalAlignment6.png", "wb") as f:
    f.write(png_data)
print("Graph visualization saved as 'graphGoalAlignment6.png'.")

#####################
#   Example Usage   #
#####################
inputs = [
    # Turn 1: Initial input provided via __start__ (or override via inputs).
    {
        "messages": [
            {"role": "user", "content": "I want to lose weight."}
        ]
    },
    # Turn 2: The user refines the goal.
    Command(resume="I want to create a personal fitness goal."),
    # Turn 3: The user adds details: combining a healthy diet and regular workouts.
    Command(resume="I plan to follow a balanced diet with reduced carbs and work out 4 times a week."),
    # Turn 4: The user provides even more details.
    Command(resume="Specifically, I want to reduce my calorie intake and incorporate cardio and strength training. I want to be 100 lbs in 2 weeks."),
    # Turn 5: The user indicates it's time to validate the goal.
    Command(resume="Ok, let's validate that goal now."),
    # Turn 6: The user gives positive feedback.
    Command(resume="Yes, I like it."),
    # Turn 7: The user expresses even stronger approval.
    Command(resume="I love it."),
    # Turn 8: The user then changes their mind about the diet portion.
    Command(resume="Actually, I've changed my mind; I don't want to focus on dieting anymore. Just let me work out."),
]

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# for idx, user_input in enumerate(inputs):
#     print(f"\n--- Conversation Turn {idx + 1} ---")
#     print(f"User Input: {user_input}\n")
#     for update in graph.stream(user_input, config=thread_config, stream_mode="updates"):
#         # Print the full update for LangSmith output
#         # print("Full update:", update)
#         # Now filter and print only messages with role "ai" or "system" 
#         # (i.e. the LLM outputs)
#         if isinstance(update, dict):
#             for node_id, val in update.items():
#                 if isinstance(val, dict) and "messages" in val:
#                     # We'll print all messages from this update that are not from the user.
#                     for msg in val["messages"]:
#                         if isinstance(msg, dict):
#                             role = msg.get("role", "")
#                             content = msg.get("content", "")
#                         else:
#                             role = getattr(msg, "role", "")
#                             content = getattr(msg, "content", "")
#                         if role in ["ai", "system"]:
#                             print(f"{node_id} ({role.upper()}): {content}")
#         else:
#             print(update)

for idx, user_input in enumerate(inputs):
    print(f"\n--- Conversation Turn {idx + 1} ---")
    print(f"User Input: {user_input}\n")
    # Print ALL messages from each update, not just the last message.

    for update in graph.stream(user_input, config=thread_config, stream_mode="updates"):
        print("Full update:", update)

        if isinstance(update, dict):
            for node_id, val in update.items():
                if isinstance(val, dict) and "messages" in val:
                    for msg in val["messages"]:
                        # Use dict access if possible; otherwise, use attribute access.
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                        else:
                            role = getattr(msg, "role", "")
                            content = getattr(msg, "content", "")
                        print(f"{node_id} ({role.upper()}): {content}")
        else:
            print(update)
