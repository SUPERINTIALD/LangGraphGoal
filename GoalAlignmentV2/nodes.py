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
    print(colorama.Fore.CYAN + ">> Entering Goal_creator_advisor \n")

    """Calls the 'goal_creator_agent' to refine or create the user's goal."""
    response = goal_creator_agent.invoke(state)
    # Check if we have a refined goal and user has confirmed
    last_messages = state.get("messages", [])
    if len(last_messages) >= 2:
        # Fix: Access content directly instead of using .get()
        last_message = last_messages[-1]
        if hasattr(last_message, "content"):
            last_user_input = last_message.content.lower()
            if "yes" in last_user_input or "confirmed" in last_user_input:
                print(colorama.Fore.GREEN + f"User confirmed the goal. Moving to node: decision_maker")
                update_goal_state(state)
                return Command(update=response, goto="decision_maker")
    
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

    if hasattr(state, "metadata") and not state.metadata.get("validated", False):
        state.metadata["validated"] = True
    elif not hasattr(state, "metadata"):
        state["metadata"] = {"validated": True}

    return Command(update=response, goto="goal_satisfied")


def human_node(
    state: MessagesState, config
) -> Command[Literal["goal_creator_advisor", "goal_validator", "goal_satisfied", "human", "decision_maker"]]:
    print(colorama.Fore.CYAN + ">> Entering human_node \n")
    triggers = config["metadata"].get("langgraph_triggers", [])
    coming_from_decision_maker = any("decision_maker" in trigger for trigger in triggers)

    user_input = interrupt(value="[Human Node] Ready for user input:")
    
    # Check for a trigger in metadata; default to goal_creator_advisor if not provided.
    if coming_from_decision_maker and user_input.strip().lower() in ["yes", "y", "confirmed", "valid"]:
        print(colorama.Fore.GREEN + "Coming from decision maker and user confirmed. Moving to node: goal_satisfied")
        return Command(
            update={"messages": [{"role": "human", "content": user_input}]},
            goto="goal_satisfied"
        )
    # if triggers and len(triggers) == 1:
    #     active_agent = triggers[0].split(":")[1]
    # else:
    #     active_agent = "goal_creator_advisor"
    active_agent = "goal_creator_advisor"
    if triggers and len(triggers) == 1:
        active_agent = triggers[0].split(":")[1]
    
    print(colorama.Fore.GREEN + f"Moving to node:"+ colorama.Fore.RED+ f"{active_agent}")
    return Command(
        update={"messages": [{"role": "human", "content": user_input}]},
        goto=active_agent,
    )


def goal_satisfied_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "decision_maker", "end_node"]]:
    print(colorama.Fore.CYAN + ">> Entering goal_satisfied_node \n")
    print(colorama.Fore.BLUE + "DEBUG: goal_satisfied_node received state: " + str(type(state)))

    # Invoke the new goal satisfied agent to assess the refined goal.
    response = goal_satisfied_agent.invoke(state)
    # Extract the output text.
    agent_text = ""
    if isinstance(response, dict) and "messages" in response:
        ai_messages = [msg.content.strip() for msg in response["messages"] if hasattr(msg, "content") and msg.content.strip()]
        agent_text = "\n".join(ai_messages).strip()
    else:
        agent_text = str(response)
    # if not agent_text:
    #     agent_text = str(response)
    print(colorama.Fore.YELLOW + f"Extracted AI message (Goal Satisfied): {agent_text}")
    print(colorama.Fore.MAGENTA + f" expecting user input")

    user_input = interrupt(value="[Goal Satisfied Node] Is this goal satisfactory? (yes/no): ")
    normalized_input = user_input.strip().lower()
    print(colorama.Fore.MAGENTA + f" confirmed user input: {normalized_input}")
    
    if any(keyword in normalized_input for keyword in ["valid", "confirmed", "yes"]):
        print(colorama.Fore.GREEN + "User confirmed the goal is valid. Moving to node: end_node")
        # final_message = {"role": "ai", "content": f"{agent_text}\nUser confirmed the goal is satisfactory."}
        # if hasattr(state, "messages"):
        #     state.messages.append(final_message)
        # else:
        #     state["messages"] = [final_message]
        # Update the goal state with the new message.
        # update_goal_state(state)
        final_message = {"role": "ai", "content": f"Great! Your confirmation on the refined goal is appreciated. Let's proceed with the project as planned. If you have any more questions or need further clarification, feel free to ask."}

        if hasattr(state, "metadata"):
            state.metadata["goal_confirmed"] = True
        else:
            state["metadata"] = {"goal_confirmed": True}
        update_goal_state(state)
        # return Command(
        #     update={"messages": [final_message]},
        #     goto="end_node"
        # )
        print(colorama.Fore.CYAN + ">> Moving to end_node via direct call \n")
        # return end_node(state)
        return Command(
            update={"messages": [{"role": "ai", "content": final_message}]},
            goto="end_node"
        )
        # return end_node(state)
    else:
        print(colorama.Fore.GREEN + "User did not confirm the goal. Moving to node: goal_creator_advisor")
        return Command(
            update={"messages": [{
                "role": "ai",
                "content": f"{agent_text}\nUser indicated further refinement is needed.",
            }]},
            goto="goal_creator_advisor"
        )
   
def decision_maker_node(
    state: MessagesState
) -> Command[Literal["goal_creator_advisor", "goal_satisfied", "end_node"]]:
    print(colorama.Fore.CYAN + ">> Entering decision_maker_node \n")
    # Invoke the decision maker agent to evaluate the goal.

    if hasattr(state, "metadata") and state["metadata"].get("goal_confirmed", False):
            print(colorama.Fore.GREEN + "Goal already confirmed. Moving directly to end_node")
            print(colorama.Fore.CYAN + ">> Moving to end_node via direct call \n")

            return Command(
                update={"messages": [{"role": "ai", "content": "Goal already confirmed, proceeding to end."}]},
                goto="end_node"
            )
    response = decision_maker_agent.invoke(state)
    
    # Extract only the AI messages that start with our marker.
    agent_text = ""
    if isinstance(response, dict) and "messages" in response:
        ai_messages = [
            msg.content.strip() 
            for msg in response["messages"]
            # if msg.get("content", "").strip().startswith("[DecisionMaker]")

            if hasattr(msg, "content") and msg.content.strip()
        ]
        agent_text = "\n".join(ai_messages).strip()
    else:
        agent_text = str(response)
    print(colorama.Fore.RED + f"Extracted AI message: {agent_text}")
    
    # Rely solely on the LLM's output.
    if any(keyword in agent_text.lower() for keyword in ["valid", "confirmed", "yes"]):
        print(colorama.Fore.GREEN + "DecisionMaker LLM determined the goal is valid. Confirm with USER:")
        
        if hasattr(state, "metadata"):
            state["metadata"]["current_goal"] = agent_text
        else:
            state["metadata"] = {"current_goal": agent_text}
            
        print(colorama.Fore.GREEN + "MOVING TO GOAL SATISFIED NODE")
        print(colorama.Fore.CYAN + ">> Moving to goal_satisfied via direct call \n")

        return goal_satisfied_node(state)
        return Command(
            update={"messages": [{"role": "ai", "content": f"{agent_text}\n Routing to goal_satisfied_node for final confirmation."}]},
            goto="goal_satisfied"
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


# Helper function to update the goal state.
def update_goal_state(state: MessagesState) -> None:
    """Updates the goal state in metadata based on the latest messages."""
    # First, ensure metadata exists using consistent dictionary access
    if not hasattr(state, "metadata"):
        state["metadata"] = {}
    
    last_messages = state.get("messages", [])
    if last_messages:
        # Find the most recent AI message with content
        for msg in reversed(last_messages):
            if hasattr(msg, "role") and msg.role == "ai" and hasattr(msg, "content"):
                # Use dictionary access for updating metadata
                state["metadata"]["current_goal"] = msg.content
                return
                
    # If no AI message found, set a default - use dictionary access
    if "current_goal" not in state["metadata"]:
        state["metadata"]["current_goal"] = "No goal defined yet."


def end_node(state: MessagesState) -> Command[Literal[END]]:
    # print(colorama.Fore.CYAN + ">> Entering end_node \n")
    
    # # Update the goal state before finalizing.
    # update_goal_state(state)
    
    # # Retrieve the stored goal ID from state.metadata.
    # goal_id = None
    # if hasattr(state, "metadata") and "goal_id" in state.metadata:
    #     goal_id = state.metadata["goal_id"]
    # elif isinstance(state, dict) and "metadata" in state and "goal_id" in state["metadata"]:
    #     goal_id = state["metadata"]["goal_id"]
    
    # if goal_id:
    #     try:
    #         from goal_manager import get_goal
    #         stored_goal = get_goal(goal_id)
    #         print(f"Final Goal (ID: {goal_id}):\n{stored_goal}")
    #     except Exception as e:
    #         print(f"Error retrieving goal: {e}")
    # else:
    #     print("No valid final goal was found in the conversation state.")

    # print("Conversation ended. Thank you for using our service!")
    # return Command(goto=END)
    print(colorama.Fore.CYAN + ">> Entering end_node \n")
    print(colorama.Fore.GREEN + "==== FINAL GOAL SUMMARY ====")
    
    if isinstance(state, dict) and "metadata" in state and "current_goal" in state["metadata"]:
        print(colorama.Fore.GREEN + f"Final Goal: {state['metadata']['current_goal']}")
    elif hasattr(state, "metadata") and hasattr(state.metadata, "current_goal"):
        print(colorama.Fore.GREEN + f"Final Goal: {state.metadata.current_goal}")
    else:
        print(colorama.Fore.YELLOW + "No final goal was found in the conversation state.")
        
    print(colorama.Fore.GREEN + "==========================")
    print(colorama.Fore.GREEN + "Conversation ended. Thank you for using our service!")
    
    # Simply return END - don't try to wrap it in a Command
    return Command(goto=END)