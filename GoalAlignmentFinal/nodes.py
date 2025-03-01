from langgraph.types import Command, interrupt
from langgraph.graph import MessagesState, END
# from functions import goal_creator_agent, goal_validator_agent
from typing import Literal
import colorama
colorama.init(autoreset=True)


from agents import goal_creator_agent, goal_validator_agent, decision_maker_agent, goal_satisfied_agent
#####################
#   Node Functions  #
#####################

def call_goal_creator_advisor(
    state: MessagesState,
) -> Command[Literal["human"]]:
    print(colorama.Fore.CYAN + ">> Entering call_goal_creator_advisor \n")

    """Calls the 'goal_creator_agent' to refine or create the user's goal."""
    response = goal_creator_agent.invoke(state)
    print(colorama.Fore.GREEN + f"Moving to node: human")

    return Command(update=response, goto="human")

def call_goal_validator(
    state: MessagesState,
) -> Command[Literal["goal_satisfied"]]:
    print(colorama.Fore.CYAN + ">> Entering call_goal_validator \n")

    """Calls the 'goal_validator_agent' to check the goal's feasibility and clarity."""
    response = goal_validator_agent.invoke(state)
    print(colorama.Fore.GREEN + f"Moving to node: goal_satisfied")

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
    # user_input = interrupt(value="LLM is satisfied. Are YOU satisfied? (yes/no):")
    # if user_input.strip().lower() in ["yes", "y"]:
    #     return Command(
    #         update={"messages": [{
    #             "role": "ai",
    #             "content": "Great! Goal is finalized.",
    #         }]},
    #         goto=END
    #     )
    # else:
    #     return Command(
    #         update={"messages": [{
    #             "role": "ai",
    #             "content": "Understood. Let's refine further.",
    #         }]},
    #         goto="goal_creator_advisor"
    #     )

# def human_node(
#     state: MessagesState, config
# ) -> Command[Literal["goal_creator_advisor", "goal_validator", "goal_satisfied", "human"]]:
#     print(colorama.Fore.CYAN + ">> Entering human_node \n")

#     """
#     A node for collecting user input.
#     We read the active agent from the trigger metadata.
#     If no trigger is provided, we default to 'goal_creator_advisor'.
#     """
#     user_input = interrupt(value="Ready for user input:")
#     triggers = config["metadata"].get("langgraph_triggers", [])
#     if len(triggers) != 1:
#         active_agent = "goal_creator_advisor"
#     else:
#         active_agent = triggers[0].split(":")[1]

#     print(colorama.Fore.GREEN + f"Moving to node: {active_agent}")

#     return Command(
#         update={"messages": [{"role": "human", "content": user_input}]},
#         goto=active_agent,
#     )
def human_node(
    state: MessagesState, config
) -> Command[Literal["goal_creator_advisor", "goal_validator", "goal_satisfied", "human", "decision_maker"]]:
    print(colorama.Fore.CYAN + ">> Entering human_node \n")
    user_input = interrupt(value="[Human Node] Ready for user input:")
    
    # Check for a trigger in metadata; default to goal_creator_advisor if not provided.
    triggers = config["metadata"].get("langgraph_triggers", [])
    if triggers and len(triggers) == 1:
        active_agent = triggers[0].split(":")[1]
    else:
        active_agent = "goal_creator_advisor"

    print(colorama.Fore.GREEN + f"Moving to node: {active_agent}")
    return Command(
        update={"messages": [{"role": "human", "content": user_input}]},
        goto=active_agent,
    )

def goal_satisfied_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "human", "decision_maker"]]:
    print(colorama.Fore.CYAN + ">> Entering goal_satisfied_node \n")
    # Invoke the new goal satisfied agent to assess the refined goal.
    response = goal_satisfied_agent.invoke(state)
    # Extract the output text.
    agent_text = ""
    if isinstance(response, dict) and "messages" in response:
        # ai_messages = [msg.content.strip() for msg in response["messages"] if hasattr(msg, "role") and msg.role == "ai"]
        ai_messages = [msg.get("content", "").strip() for msg in response["messages"] if msg.get("role") == "ai"]

        agent_text = "\n".join(ai_messages).strip()
    else:
        agent_text = str(response)
    print(colorama.Fore.RED + f"Extracted AI message (Goal Satisfied): {agent_text}")
    
    # Check if the agent has indicated satisfaction.
    if "confirmed" in agent_text.lower():
        print(colorama.Fore.GREEN + "Goal satisfied confirmed. Moving to node: decision_maker")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nThe goal is confirmed as satisfactory.",
            }]},
            goto="decision_maker"
        )
    else:
        print(colorama.Fore.GREEN + "Goal not confirmed. Moving to node: goal_creator_advisor")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nThe goal is not yet fully satisfactory. Let's refine it further.",
            }]},
            goto="goal_creator_advisor"
        )
# def goal_satisfied_node(
#     state: MessagesState
# ) -> Command[Literal["goal_creator_advisor", "human", "decision_maker"]]:
#     # Check if LLM is satisfied (looking for 'valid' in last AI message).
#     print(colorama.Fore.CYAN + ">> Entering goal_satisfied_node \n")

#     llm_satisfied = False
#     if state.messages:
#         last_ai_msg = [m for m in state.messages if m["role"] == "ai"]
#         if last_ai_msg and "valid" in last_ai_msg[-1]["content"].lower():
#             llm_satisfied = True

#     if not llm_satisfied:
#         print(colorama.Fore.GREEN + "Moving to node: goal_creator_advisor")

#         return Command(
#             update={"messages": [{
#                 "role": "ai",
#                 "content": "The LLM is NOT satisfied with the goal yet. Let's refine more.",
#             }]},
#             goto="goal_creator_advisor"
#         )
#     print(colorama.Fore.GREEN + "Moving to node: decision_maker")

#     # If LLM is satisfied, move to the decision maker node.
#     return Command(
#         update={"messages": [{
#             "role": "ai",
#             "content": "The LLM is satisfied with the goal. Now let's confirm with you.",
#         }]},
#         goto="decision_maker"
#     )
# def decision_maker_node(
#     state: MessagesState
# ) -> Command[Literal["goal_creator_advisor", "human", "end_node"]]:
#     print(colorama.Fore.CYAN + ">> Entering decision_maker_node \n")

#     """
#     Uses the decision maker agent to evaluate the current goal, then presents a 
#     human-friendly summary to the user for final confirmation. If the user confirms 
#     the goal is satisfactory, the flow proceeds to finalization; otherwise, it returns 
#     to goal refinement.
#     """
#     # Invoke the decision maker agent to get its analysis of the current goal
#     response = decision_maker_agent.invoke(state)
    
#     # Extract the agent's output text from its response (assuming it returns a dict with a "messages" list)
#     agent_text = ""
#     if isinstance(response, dict) and "messages" in response:
#         # Collect all AI messages into a single summary
#         ai_messages = [msg.content for msg in response["messages"] if hasattr(msg, "role") and msg.role == "ai"]
#         agent_text = "\n".join(ai_messages).strip()
#     else:
#         agent_text = str(response)
    
#     # Create a friendly summary message for the user
#     friendly_message = (
#         "Based on my analysis of your goal, here's what I understand:\n"
#         f"{agent_text}\n\n"
#         "Does this summary reflect your intended goal? Please answer 'yes' if you're satisfied, "
#         "or 'no' if you'd like to refine it further."
#     )
    
#     # Ask the user for final confirmation using a clear, friendly prompt
#     user_input = interrupt(value="Final decision: Are you satisfied with your goal? (yes/no):")
    
#     if user_input.strip().lower() in ["yes", "y"]:
#         print(colorama.Fore.GREEN + "Moving to node: end_node")

#         return Command(
#             update={"messages": [{
#                 "role": "ai",
#                 "content": f"Great! You confirmed that your goal is satisfactory. We'll finalize it now.\n\n{friendly_message}",
#             }]},
#             goto="end_node"
#         )
#     else:
#         print(colorama.Fore.GREEN + "Moving to node: goal_creator_advisor")

#         return Command(
#             update={"messages": [{
#                 "role": "ai",
#                 "content": f"Understood. Let's refine your goal further. Here is what I gathered so far:\n\n{friendly_message}",
#             }]},
#             goto="goal_creator_advisor"
#         )



def decision_maker_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "end_node"]]:
    print(colorama.Fore.CYAN + ">> Entering decision_maker_node \n")
    # Invoke the decision maker agent to evaluate the goal.
    response = decision_maker_agent.invoke(state)
    
    # Extract only the AI messages that start with our marker.
    agent_text = ""
    if isinstance(response, dict) and "messages" in response:
        ai_messages = [
            msg.content.strip() 
            for msg in response["messages"]
            # if msg.get("content", "").strip().startswith("[DecisionMaker]")

            if hasattr(msg, "content") and msg.content.strip().startswith("[DecisionMaker]")
        ]
        agent_text = "\n".join(ai_messages).strip()
    else:
        agent_text = str(response)
    print(colorama.Fore.RED + f"Extracted AI message: {agent_text}")
    
    # Rely solely on the LLM's output.
    # For example, if the message contains "valid", "confirmed" or "yes", we move forward.
    if any(keyword in agent_text.lower() for keyword in ["valid", "confirmed", "yes"]):
        print(colorama.Fore.GREEN + "DecisionMaker LLM determined the goal is valid. Moving to node: end_node")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nThe decision maker LLM confirmed the goal is satisfactory.",
            }]},
            goto="end_node"
        )
    else:
        print(colorama.Fore.GREEN + "DecisionMaker LLM determined the goal is not valid. Moving to node: goal_creator_advisor")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nThe decision maker LLM determined the goal is not yet satisfactory. Refinement is needed.",
            }]},
            goto="goal_creator_advisor"
        )
# def decision_maker_node(
#     state: MessagesState
# ) -> Command[Literal["goal_creator_advisor", "human", "end_node"]]:
#     print(colorama.Fore.CYAN + ">> Entering decision_maker_node \n")

#     # Ask the user for a final decision.
#     response = decision_maker_agent.invoke(state)
#     # print("Decision Maker Response:", response)

#       # Debug: print out each message in the response
#     # if isinstance(response, dict) and "messages" in response:
#     #     for idx, msg in enumerate(response["messages"]):
#     #         # Use .strip() to clean up any extra whitespace
#     #         content = msg.content.strip() if hasattr(msg, "content") else ""
#     #         role = msg.role if hasattr(msg, "role") else "unknown"
#     #         print(colorama.Fore.YELLOW + f"DEBUG: Message {idx} | Role: {role} | Content: {content}")
    
#     # Extract only the AI messages that start with our marker, after stripping whitespace
#     agent_text = ""
#     if isinstance(response, dict) and "messages" in response:
#         ai_messages = [
#             msg.content.strip() 
#             for msg in response["messages"]
#             # if msg.get("content", "").strip().startswith("[DecisionMaker]")

#             if hasattr(msg, "content") and msg.content.strip().startswith("[DecisionMaker]")
#         ]
#         agent_text = "\n".join(ai_messages).strip()
#     else:
#         agent_text = str(response)
    
#     print(colorama.Fore.RED + f"Extracted AI message: {agent_text}")
#     # user_input = interrupt(value="Final decision: Are you satisfied with your goal? (yes/no):")
#     user_input = interrupt(value="[Decision Maker Node] Final decision: Are you satisfied with your goal? (yes/no):")

#     if user_input.strip().lower() in ["yes", "y"]:
#         print(colorama.Fore.GREEN + "Moving to node: end_node")

#         return Command(

#             update={"messages": [{
#                 "role": "ai",
#                 "content": f"{response}\nUser confirmed that the goal is satisfactory.",
#             }]},
#             goto="end_node"
#         )
#     else:
#         print(colorama.Fore.GREEN + "Moving to node: goal_creator_advisor")

#         return Command(

#             update={"messages": [{
#                 "role": "ai",
#                 "content": f"{response}\nUnderstood, let's refine the goal further.",
#             }]},
#             goto="goal_creator_advisor"
#         )


def end_node(state: MessagesState) -> Command[Literal[END]]:
    print(colorama.Fore.CYAN + ">> Entering end_node \n")

    """
    Finalizes the conversation by extracting the current goal from the conversation
    and storing it using the goal manager. This avoids keeping the full MessageState.
    """
    # Dynamically extract the most recent AI output that represents the refined goal.
    final_goal = None
    for message in reversed(state.messages):
        # You may want to adjust this extraction logic if your protocol differs.
        if message.get("role") == "ai" and message.get("content"):
            final_goal = message["content"].strip()
            break

    if final_goal:
        try:
            # Import and store the goal without any hardcoding
            from goal_manager import store_goal
            goal_id = store_goal(final_goal)
            print(f"Goal successfully stored with ID: {goal_id}")
        except Exception as e:
            print(f"Error storing goal: {e}")
    else:
        print("No valid final goal was found in the conversation state.")

    print("Conversation ended. Thank you for using our service!")
    return Command(goto=END)