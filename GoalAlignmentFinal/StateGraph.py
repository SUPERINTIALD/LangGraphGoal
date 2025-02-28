from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from nodes import call_goal_creator_advisor, call_goal_validator, goal_satisfied_node, human_node, goal_satisfied_node, decision_maker_node, end_node
#####################
#   Build the Graph #
#####################

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("goal_creator_advisor", call_goal_creator_advisor)
builder.add_node("goal_validator", call_goal_validator)
builder.add_node("goal_satisfied", goal_satisfied_node)
builder.add_node("human", human_node)
builder.add_node("decision_maker", decision_maker_node)
builder.add_node("end_node", end_node)

# Edges
builder.add_edge(START, "goal_creator_advisor")
builder.add_edge("goal_creator_advisor", "human")
builder.add_edge("goal_validator", "goal_satisfied")
builder.add_edge("goal_satisfied", "goal_creator_advisor")
builder.add_edge("goal_satisfied", "decision_maker")
builder.add_edge("decision_maker", "goal_creator_advisor")
builder.add_edge("goal_satisfied", "end_node")


checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)