
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# from nodes import call_goal_creator_advisor, call_goal_validator, goal_satisfied_node, human_node, goal_satisfied_node, decision_maker_node, end_node
from tools import dummy_tool
# from StateGraph import builder
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
