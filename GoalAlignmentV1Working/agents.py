
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import dummy_tool
#####################
#   Agents (LLMs)   #
#####################

goal_creator_model = ChatOpenAI(model="gpt-4-mini", temperature=0.7)
goal_validator_model = ChatOpenAI(model="gpt-4-mini", temperature=0.0)
decision_maker_model = ChatOpenAI(model="gpt-4-mini", temperature=0.7)


goal_creator_tools = [dummy_tool]
goal_creator_agent = create_react_agent(
    goal_creator_model,
    goal_creator_tools,
    prompt=(
        "You are a 'goal creation advisor'. "
        "You help the user refine their goal statement. "
        "If you need extra details or clarifications, call the 'dummy_tool' or ask the user. "
        "Always provide a short, clear statement of the refined goal in your final message."
        ""
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


decision_maker_tools = [dummy_tool]
# decision_maker_agent = create_react_agent(
#     decision_maker_model,
#     decision_maker_tools,
#      prompt=(
#         "You are a 'goal decision maker router'. Begin every response with [DecisionMaker]. "
#         "Analyze the current state of the goal and determine if it is fully refined, feasible, and clear. "
#         "If you believe the goal still needs work, ask for more clarification or details. "
#         "If the goal appears valid, include the word 'valid' in your final message. "
#         "After your analysis, prompt the user to confirm if they agree with your assessment."
#     ),
# )

decision_maker_agent = create_react_agent(
    decision_maker_model,
    decision_maker_tools,
    prompt=(
        "You are a 'goal decision maker router'. Begin every response with [DecisionMaker]. "
        "Analyze the current state of the goal and determine if it is fully refined, feasible, and clear. "
        "If you believe the goal still needs work, ask for more clarification or details. "
        "Only move forward if the goal is fully refined, feasible, and clear, and the user agrees. "
        "If the goal appears valid, include the word 'valid' in your final message."
    ),
)

goal_satisfied_model = ChatOpenAI(model="gpt-4", temperature=0.7)
goal_satisfied_tools = [dummy_tool]
goal_satisfied_agent = create_react_agent(
    goal_satisfied_model,
    goal_satisfied_tools,
    prompt=(
        "You are a 'goal satisfied agent'. Your task is to review the refined goal and determine if "
        "both the LLM and the user appear satisfied with it. "
        "Always ask the user if they are satisfied with the goal. "
        "If the user is satisfied, include the word 'confirmed' in your final message. "

    ),
)