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

from StateGraph import graph
# from nodes import call_goal_creator_advisor, call_goal_validator, goal_satisfied_node, human_node, goal_satisfied_node, decision_maker_node, end_node
# from tools import dummy_tool
# from StateGraph import builder
# from agents import 
#####################
#   Example Usage   #
#####################
# inputs = [
#     # Turn 1: Initial input provided via __start__ (or override via inputs).
#     {
#         "messages": [
#             {"role": "user", "content": "I want to lose weight."}
#         ]
#     },
#     # Turn 2: The user refines the goal.
#     Command(resume="I want to create a personal fitness goal."),
#     # Turn 3: The user adds details: combining a healthy diet and regular workouts.
#     Command(resume="I plan to follow a balanced diet with reduced carbs and work out 4 times a week."),
#     # Turn 4: The user provides even more details.
#     Command(resume="Specifically, I want to reduce my calorie intake and incorporate cardio and strength training. I want to be 100 lbs in 2 weeks."),
#     # Turn 5: The user indicates it's time to validate the goal.
#     Command(resume="Ok, let's validate that goal now."),
#     # Turn 6: The user gives positive feedback.
#     Command(resume="Yes, I like it."),
#     # Turn 7: The user expresses even stronger approval.
#     Command(resume="I love it."),
#     # Turn 8: The user then changes their mind about the diet portion.
#     Command(resume="Actually, I've changed my mind; I don't want to focus on dieting anymore. Just let me work out."),
# ]
try:
    mermaid_syntax = graph.get_graph().draw_mermaid()
    print("Generated Mermaid Syntax:\n")
    print(mermaid_syntax)  # Print the Mermaid syntax for debugging

    # Now attempt to render the PNG
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graphGoalAlignmentHumanVer1.png", "wb") as f:
        f.write(png_data)
    print("Graph visualization saved as 'graphGoalAlignmentHumanVer1.png'.")
except Exception as e:
    print(f"Error generating graph visualization: {e}")

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

while True:
    user_text = input("Enter your message (or type 'exit' to quit): ")
    if user_text.lower() in ["exit", "quit"]:
        break

    # Create a user message dictionary to pass to the graph.
    user_input = {"messages": [{"role": "user", "content": user_text}]}
    
    for update in graph.stream(user_input, config=thread_config, stream_mode="updates"):
        if isinstance(update, dict):
            for node_id, val in update.items():
                if isinstance(val, dict) and "messages" in val:
                    for msg in val["messages"]:
                        # Retrieve role and content robustly.
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                        else:
                            role = getattr(msg, "role", "")
                            content = getattr(msg, "content", "")
                        # Print output messages (skip human echoes if needed)
                        if role != "human":
                            print(f"{node_id} ({role.upper()}): {content}")
        else:
            print(update)
