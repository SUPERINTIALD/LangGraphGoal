from langgraph.types import Command, interrupt
from langgraph.graph import MessagesState, END
# from functions import goal_creator_agent, goal_validator_agent
from typing import Literal


from agents import goal_creator_agent, goal_validator_agent
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

# def goal_satisfied_node(
#     state: MessagesState
# ) -> Command[Literal["goal_creator_advisor", "human", END]]:
#     """
#     Checks if both LLM and user are satisfied. 
#     If not, routes back to creation; if user says yes, goes to END.
#     """
#     # 1) Check if LLM is satisfied (looking for 'valid' in last AI message).
#     llm_satisfied = False
#     if state.messages:
#         last_ai_msg = [m for m in state.messages if m["role"] == "ai"]
#         if last_ai_msg and "valid" in last_ai_msg[-1]["content"].lower():
#             llm_satisfied = True

#     # 2) If LLM not satisfied, route back to the goal creator.
#     if not llm_satisfied:
#         return Command(
#             update={"messages": [{
#                 "role": "ai",
#                 "content": "The LLM is NOT satisfied with the goal yet. Let's refine more.",
#             }]},
#             goto="goal_creator_advisor"
#         )

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

def goal_satisfied_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "human", "decision_maker"]]:
    # Check if LLM is satisfied (looking for 'valid' in last AI message).
    llm_satisfied = False
    if state.messages:
        last_ai_msg = [m for m in state.messages if m["role"] == "ai"]
        if last_ai_msg and "valid" in last_ai_msg[-1]["content"].lower():
            llm_satisfied = True

    if not llm_satisfied:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "The LLM is NOT satisfied with the goal yet. Let's refine more.",
            }]},
            goto="goal_creator_advisor"
        )
    
    # If LLM is satisfied, move to the decision maker node.
    return Command(
        update={"messages": [{
            "role": "ai",
            "content": "The LLM is satisfied with the goal. Now let's confirm with you.",
        }]},
        goto="decision_maker"
    )

def decision_maker_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "human", "end_node"]]:
    # Ask the user for a final decision.
    user_input = interrupt(value="Final decision: Are you satisfied with your goal? (yes/no):")
    if user_input.strip().lower() in ["yes", "y"]:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "Great! Your goal is finalized.",
            }]},
            goto="end"
        )
    else:
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": "Understood. Let's refine your goal further.",
            }]},
            goto="goal_creator_advisor"
        )


def end_node(state: MessagesState) -> Command[Literal[END]]:
    print("Conversation ended. Thank you for using our service!")
    return Command(goto=END)