import uuid
from dotenv import load_dotenv
load_dotenv()
import colorama
colorama.init(autoreset=True)
from StateGraph import graph
###############
# PNG DIAGRAM #
###############
try:
    mermaid_syntax = graph.get_graph().draw_mermaid()
    print("Generated Mermaid Syntax:\n")
    print(mermaid_syntax) 
    png_data = graph.get_graph().draw_mermaid_png()
    with open("./png/graphGoalAlignmentHumanVer1.png", "wb") as f:
        f.write(png_data)
    print("Graph visualization saved as 'graphGoalAlignmentHumanVer1.png'.")
except Exception as e:
    print(f"Error generating graph visualization: {e}")

################
# USER INTERACTION #
################
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

while True:
    user_text = input(colorama.Fore.MAGENTA + "Enter your message (or type 'exit' to quit): ")
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
