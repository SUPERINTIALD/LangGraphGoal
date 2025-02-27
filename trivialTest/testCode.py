from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

# Create an instance of ChatOpenAI.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a conversation memory to store messages.
conversation_memory = ChatMessageHistory()

def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the LLM, records the prompt and response,
    and returns the response.
    """
    response = llm.invoke(prompt)
    conversation_memory.add_user_message(prompt)
    conversation_memory.add_ai_message(response)
    return response

def conversation_loop():
    """
    Runs an interactive multi-turn conversation.
    The conversation starts with an initial prompt and then waits for user input via Python's input().
    Type "exit" to quit.
    """
    # Initial prompt
    initial_prompt = "Hello, how are you today?"
    print("User:", initial_prompt)
    response = ask_llm(initial_prompt)
    print("LLM:", response)
    
    # Interactive loop using input() for user responses.
    while True:
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break
        print("User:", user_input)
        response = ask_llm(user_input)
        print("LLM:", response)
    
    # Print conversation history at the end.
    print("\n--- Conversation History ---")
    messages = conversation_memory.get_messages()
    for message in messages:
        print(message)

if __name__ == "__main__":
    conversation_loop()
