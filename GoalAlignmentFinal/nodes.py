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
    update_goal_state(state)

    return Command(update=response, goto="human")

def call_goal_validator(
    state: MessagesState,
) -> Command[Literal["goal_satisfied"]]:
    print(colorama.Fore.CYAN + ">> Entering call_goal_validator \n")

    """Calls the 'goal_validator_agent' to check the goal's feasibility and clarity."""
    response = goal_validator_agent.invoke(state)
    print(colorama.Fore.GREEN + f"Moving to node: goal_satisfied")

    return Command(update=response, goto="goal_satisfied")

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
) -> Command[Literal["goal_creator_advisor", "goal_validator", "goal_satisfied", "human", "decision_maker", "end_node"]]:
    print(colorama.Fore.CYAN + ">> Entering human_node \n")
    user_input = interrupt(value="[Human Node] Ready for user input:")
    
    # Check for a trigger in metadata; default to goal_creator_advisor if not provided.
    triggers = config["metadata"].get("langgraph_triggers", [])
    if triggers and len(triggers) == 1:
        active_agent = triggers[0].split(":")[1]
    else:
        active_agent = "goal_creator_advisor"

    print(colorama.Fore.GREEN + f"Moving to node:"+ colorama.Fore.RED+ {active_agent})
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
        ai_messages = [msg.content.strip() for msg in response["messages"] if hasattr(msg, "content") and msg.content.strip()]

        agent_text = "\n".join(ai_messages).strip()
    else:
        agent_text = str(response)
    if not agent_text:
        agent_text = str(response)
    print(colorama.Fore.YELLOW + f"Extracted AI message (Goal Satisfied): {agent_text}")
    
    print(colorama.Fore.MAGENTA + f" expecting user input")
    # normalized_input = user_input.strip().lower()

    print(colorama.Fore.MAGENTA + f" confirmed user input")
    if any(keyword in agent_text.lower() for keyword in ["valid", "confirmed", "yes"]) :

        print(colorama.Fore.GREEN + "(Goal Satisfied) LLM determined the user confirmed is valid. Moving to node: end_node")
        final_message = {"role": "ai", "content": f"{agent_text}\nThe decision maker LLM confirmed the goal is satisfactory."}
        if hasattr(state, "messages"):
            state.messages.append(final_message)
        else:
            state.setdefault("messages", []).append(final_message)
        # Update the goal state with the new message.
        update_goal_state(state)
        return end_node(state)
    # if any(keyword in user_input.strip().lower() for keyword in ["valid", "confirmed", "yes"]):
    #     print(colorama.Fore.GREEN + "User confirmed the goal satisfied assessment. Moving to node: end_node")
    #     final_message = {"role": "ai", "content": f"{agent_text}\nUser confirmed the goal is satisfactory."}
    #     if hasattr(state, "messages"):
    #         state.messages.append(final_message)
    #     else:
    #         state.setdefault("messages", []).append(final_message)
    #     update_goal_state(state)
    #     if hasattr(state, "metadata"):
    #         state.metadata["goal_confirmed"] = True
    #     else:
    #         state["metadata"] = {"goal_confirmed": True}
    #     return end_node(state)
    else:
        print(colorama.Fore.GREEN + "User did not confirm the goal satisfied assessment. Routing to node: decision_maker_node")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nUser indicated further refinement is needed.",
            }]},
            goto="decision_maker_node"
        )
    #     return end_node(state)
    # Check if the agent has indicated satisfaction.
    # if "confirmed" in agent_text.lower():
    #     print(colorama.Fore.GREEN + "Goal satisfied confirmed. Moving to node: decision_maker")
    #     return Command(
    #         update={"messages": [{
    #             "role": "ai",
    #             "content": f"{agent_text}\nThe goal is confirmed as satisfactory.",
    #         }]},
    #         goto="decision_maker"
    #     )
    # else:
    # print(colorama.Fore.GREEN + "Goal not confirmed. Moving to node: decision_maker_node")
    # return Command(
    #     update={"messages": [{
    #         "role": "ai",
    #         "content": f"{agent_text}\n(Proceeding to decision maker node for final evaluation.)",
    #     }]},
    #     goto="decision_maker_node"
    # )

def decision_maker_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "end_node"]]:
    print(colorama.Fore.CYAN + ">> Entering decision_maker_node \n")
    # Invoke the decision maker agent to evaluate the goal.

    if hasattr(state, "metadata") and state.metadata.get("goal_confirmed", False):
            print(colorama.Fore.GREEN + "Goal already confirmed. Moving directly to end_node")
            return end_node(state)
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
        print(colorama.Fore.GREEN + "DecisionMaker LLM determined the goal is valid. Confirm with USER:")
        print(colorama.Fore.GREEN + "MOVING TO GOAL SATISFIED NODE")

        # return Command(
        #     update={"messages": [{"role": "ai", "content": f"{agent_text}\n Routing to goal_satisfied_node for final confirmation."}]},
        #     goto="goal_satisfied"
        # )
        return goal_satisfied_node(state)

        # final_message = {"role": "ai", "content": f"{agent_text}\nThe decision maker LLM confirmed the goal is satisfactory."}
        # user_input = interrupt(value="Decision Maker: Do you agree with the above assessment? (yes/no):")
    
        # if user_input.strip().lower() in ["yes", "y"]:
        #     print(colorama.Fore.GREEN + "User confirmed the goal. Moving to node: end_node")
            
        #     if hasattr(state, "messages"):
        #         state.messages.append(final_message)
        #     else:
        #         state.setdefault("messages", []).append(final_message)
        #     # Update the goal state with the new message.
        #     update_goal_state(state)
        #         # return Command(
        #         #     update={"messages": [{
        #         #         "role": "ai",
        #         #         "content": f"{agent_text}\nUser confirmed that the goal is satisfactory.",
        #         #     }]},
        #         #     goto="end_node"
        #         # )
        #     return end_node(state)


        # else:
        #     print(colorama.Fore.GREEN + "User did not confirm the goal. Moving to node: goal_creator_advisor")
        #     return Command(
        #         update={"messages": [{
        #             "role": "ai",
        #             "content": f"{agent_text}\nUser indicated the goal is not satisfactory. Let's refine it further.",
        #         }]},
        #         goto="goal_creator_advisor"
        #     )
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



# Helper function to update the goal state.
def update_goal_state(state: MessagesState) -> None:
    """
    Extracts the most recent AI message (the refined goal) from the state,
    and then either stores or updates it using the goal_manager.
    The goal ID is saved in state.metadata (or state["metadata"]) for later retrieval.
    """
    # Retrieve messages from state (supporting both attribute and dict style).
    messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
    
    # Extract the most recent AI message.
    final_goal = None
    for message in reversed(messages):
        # Use .get() if available; otherwise, fallback to attribute access.
        if hasattr(message, "get"):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            
        if role == "ai" and content:
            final_goal = content.strip()
            break

    if not final_goal:
        print("No refined goal found to store/update.")
        return

    # Use a metadata dictionary to hold our goal ID.
    # First, check if state already has metadata; if not, create it.
    if hasattr(state, "metadata"):
        metadata = state.metadata
    else:
        metadata = state.get("metadata", {})
    
    # If a goal_id is already present, update the goal.
    if "goal_id" in metadata:
        try:
            from goal_manager import update_goal
            update_goal(metadata["goal_id"], final_goal)
            print(f"Goal updated with ID: {metadata['goal_id']}")
        except Exception as e:
            print(f"Error updating goal: {e}")
    else:
        # Otherwise, store the new goal.
        try:
            from goal_manager import store_goal
            goal_id = store_goal(final_goal)
            # Save the goal_id back into state metadata.
            if hasattr(state, "metadata"):
                state.metadata["goal_id"] = goal_id
            else:
                metadata["goal_id"] = goal_id
                state["metadata"] = metadata
            print(f"Goal successfully stored with ID: {goal_id}")
        except Exception as e:
            print(f"Error storing goal: {e}")

def end_node(state: MessagesState) -> Command[Literal[END]]:
    print(colorama.Fore.CYAN + ">> Entering end_node \n")
    
    # Update the goal state before finalizing.
    update_goal_state(state)
    
    # Retrieve the stored goal ID from state.metadata.
    goal_id = None
    if hasattr(state, "metadata") and "goal_id" in state.metadata:
        goal_id = state.metadata["goal_id"]
    elif isinstance(state, dict) and "metadata" in state and "goal_id" in state["metadata"]:
        goal_id = state["metadata"]["goal_id"]
    
    if goal_id:
        try:
            from goal_manager import get_goal
            stored_goal = get_goal(goal_id)
            print(f"Final Goal (ID: {goal_id}):\n{stored_goal}")
        except Exception as e:
            print(f"Error retrieving goal: {e}")
    else:
        print("No valid final goal was found in the conversation state.")

    print("Conversation ended. Thank you for using our service!")
    return Command(goto=END)