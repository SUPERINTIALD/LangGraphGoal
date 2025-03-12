import re
import json
import requests
from typing import List, Optional, Union, Any, Mapping
from langchain.schema import Generation, LLMResult, AgentAction, AgentFinish
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents.agent import AgentOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.agents.agent import AgentOutputParser

from pydantic import Field

class DeepSeekLLM(BaseChatModel):
    """
    Custom LangChain wrapper for DeepSeek R1 API compatible with LangGraph.
    """
    api_url: str = Field(..., description="The URL of the DeepSeek API.")
    model_name: str = Field(..., description="The name of the model to use.")
    temperature: float = 0.7
    show_thinking: bool = False
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:    
        """
        Calls the DeepSeek API and returns a cleaned response.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code}, {response.text}")

        # Process the API's streamed JSON response
        raw_response = response.text.strip()
        assistant_response = ""
        for line in raw_response.splitlines():
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    # Extract "Answer" or "Final Answer" explicitly if available
                    if "**Answer:**" in content:
                        return content.split("**Answer:**")[-1].strip()
                    if "<think>" in content and "</think>" in content:
                        if self.show_thinking:
                            # If show_thinking is True, keep the thinking but remove the tags
                            content = content.replace("<think>", "[Thinking] ").replace("</think>", " [/Thinking]")
                        else:
                            # If show_thinking is False, remove the thinking sections entirely
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                            content = content.replace("<think>", "").replace("</think>", "")

                    # if "<think>" in content or "</think>" in content:
                    #     content = content.replace("<think>", "").replace("</think>", "").strip()

                    assistant_response += content

            except json.JSONDecodeError:
                continue
        if not self.show_thinking:
            assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
            assistant_response = assistant_response.replace("<think>", "").replace("</think>", "")
            
        return assistant_response

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, **kwargs
    ) -> ChatResult:
        """
        Process messages and generate a response compatible with LangGraph.
        """
        # Convert LangChain messages to a prompt format DeepSeek can understand
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n\n"
            else:
                prompt += f"{message.type}: {message.content}\n\n"
        
        # Call the API
        response_text = self._call(prompt, stop, **kwargs)
        
        # Create an AIMessage with the response
        message = AIMessage(content=response_text)
        
        # Create a ChatGeneration with the message
        generation = ChatGeneration(message=message)
        
        # Return a ChatResult with the generation
        return ChatResult(generations=[generation])

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, **kwargs
    ) -> ChatResult:
        """
        Async version - just calls the sync version for now
        """
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
    
    def bind_tools(self, tools_classes):
        """Support for tool binding"""
        return self
    
    # LangGraph compatibility methods
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters for serialization."""
        return {"model_name": self.model_name}

# class DeepSeekLLM(BaseLLM):
#     """
#     Custom LangChain wrapper for DeepSeek R1 API.
#     """
#     api_url: str = Field(..., description="The URL of the DeepSeek API.")
#     model_name: str = Field(..., description="The name of the model to use.")

#     def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:    
#         """
#         Calls the DeepSeek API and returns a cleaned response.
#         """
#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ]
#         }

#         response = requests.post(self.api_url, json=payload)

#         if response.status_code != 200:
#             raise Exception(f"DeepSeek API error: {response.status_code}, {response.text}")

#         # Process the API's streamed JSON response
#         raw_response = response.text.strip()
#         assistant_response = ""
#         for line in raw_response.splitlines():
#             try:
#                 data = json.loads(line)
#                 if "message" in data and "content" in data["message"]:
#                     content = data["message"]["content"]
#                     # Extract "Answer" or "Final Answer" explicitly if available
#                     if "**Answer:**" in content:
#                         return content.split("**Answer:**")[-1].strip()
#                     if "<think>" in content or "</think>" in content:

#                         content = content.replace("<think>", "").replace("</think>", "").strip()

#                     assistant_response += content
#                     # assistant_response += content.strip()

#             except json.JSONDecodeError:
#                 continue

#         return assistant_response


#     async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
#         """
#         Asynchronous method to handle prompts in a batch.
#         """
#         return self._generate(prompts, stop, **kwargs)

#     def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
#         """
#         Handles batch processing of prompts and formats the output as an LLMResult.
#         """
#         generations = []
#         for prompt in prompts:
#             output = self._call(prompt, stop=stop, **kwargs)
#             generations.append([Generation(text=output)])

#         return LLMResult(generations=generations)

#     @property
#     def _llm_type(self) -> str:
#         return "deepseek"
    
#     def bind_tools(self, tools_classes):
#         return self
    
#     def invoke(self, prompt: str, **kwargs):
#         text = self._call(prompt, **kwargs)
#         # Create a dummy response object that can have attributes assigned.
#         response = type("LLMResponse", (), {})()
#         response.text = text
#         return response

class DeepSeekOutputParser(AgentOutputParser):
    """
    Custom output parser to handle DeepSeek LLM responses.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the LLM's output to determine the next action or the final answer.
        """
        # Attempt to find "Final Answer" or "Answer"
        if "**Answer:**" in text:
            final_answer = text.split("**Answer:**")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        # Look for Action and Input
        match = re.search(r"Action: (.*?)\nAction Input: (.*)", text, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=text)

        # Handle unstructured or verbose outputs
        # For now, we'll treat the entire text as the final answer
        return AgentFinish(return_values={"output": text.strip()}, log=text)

    @property
    def _type(self) -> str:
        return "deepseek_parser"


# Step 1: Replace ChatOpenAI with DeepSeekLLM
api_url = "http://localhost:11434/api/chat"
deepseek_model = "deepseek-r1:14b"

# Initialize DeepSeek LLM
llm = DeepSeekLLM(api_url=api_url, model_name=deepseek_model)

# Step 2: Add Memory
memory = ConversationBufferMemory(k=3)

# Step 3: Define Tools
def llm_query_tool(input_text: str) -> str:
    """A simple tool that queries the LLM."""
    return llm(input_text)

query_tool = Tool(
    name="DeepSeek Query Tool",
    func=llm_query_tool,
    description="Use this tool to query the DeepSeek LLM with questions that cannot be solved with Tools."
)





def add_numbers(input_text: str) -> str:
    """
    Extracts numbers and computes their sum.
    """
    numbers = list(map(int, re.findall(r'-?\d+', input_text)))  # Handle negatives
    if not numbers:
        return "No numbers found to add."
    return f"The sum is: {sum(numbers)}"


def reverse_string(input_text: str) -> str:
    """
    Reverses the input string.
    """
    if not input_text.strip():
        return "No string found to reverse."
    return f"The reversed string is: {input_text[::-1]}"


def weather_lookup(input_text: str) -> str:
    """
    Simulates a weather lookup tool with improved location extraction.
    """
    # Extract location from input, using common phrases about weather
    # location_match = re.search(
    #     r'(?:weather(?: in| like in)?|forecast for|conditions in) ([\w\s]+)', input_text, re.IGNORECASE
    # )
    # location = location_match.group(1).strip() if location_match else "New York"

    # Simulated weather data for demonstration purposes
    location = input_text
    weather_data = {
        "temperature": 18,
        "condition": "cloudy",
        "humidity": 70,
        "wind_speed": 15,
    }
    return f"The current weather in {location} is {weather_data['condition']} with a temperature of {weather_data['temperature']}°C, humidity at {weather_data['humidity']}%, and wind speed of {weather_data['wind_speed']} km/h.".format()
    # return (f"The current weather in {location} is {weather_data['condition']} "
    #         f"with a temperature of {weather_data['temperature']}°C, "
    #         f"humidity at {weather_data['humidity']}%, and wind speed of "
    #         f"{weather_data['wind_speed']} km/h.")


# Create tools
add_tool = Tool(
    name="add_tool",
    func=add_numbers,
    description=(
        "Use this tool for any addition task. Input should be numbers in a format like '45 + 20'. Don't include commas in the input"
        "Do not attempt to compute directly; always rely on this tool."
    ),
)

reverse_tool = Tool(
    name="reverse_tool",
    func=reverse_string,
    description=(
        "Use this tool to reverse a string. Input should be the string to reverse."
    ),
)

weather_tool = Tool(
    name="weather_tool",
    func=weather_lookup,
    description=(
        "Use this tool to fetch weather information. Input should be the location name."
    ),
)
def test_tools():
    assert add_numbers("10 + 20") == "The sum is: 30"
    assert add_numbers("Add -5 and 15") == "The sum is: 10"
    assert add_numbers("Nothing to add here") == "No numbers found to add."

    assert reverse_string("DeepSeek") == "The reversed string is: keeSpeeD"
    assert reverse_string("") == "No string found to reverse."

    # assert weather_lookup("What's the weather in London?") == (
    #     "The current weather in London is sunny with a temperature of 25°C, "
    #     "humidity at 60%, and wind speed of 10 km/h."
    # )
    # assert weather_lookup("Tell me the weather") == (
    #     "The current weather in New York is sunny with a temperature of 25°C, "
    #     "humidity at 60%, and wind speed of 10 km/h."
    # )

    print("All tool tests passed!")

# Step 4: Create the Prompt Template
tools = [query_tool, add_tool, reverse_tool, weather_tool]

# tools = [add_tool, reverse_tool, weather_tool]

# prompt = PromptTemplate(
#     input_variables=["input", "agent_scratchpad"],
#     template=(
#         "Answer the following question as accurately as possible. "
#         "Please provide a structured response with 'Final Answer: [answer]'. "
#         "If you need to use a tool, specify 'Action: [tool_name]' and 'Action Input: [input]'.\n\n"
#         "Question: {input}\n{agent_scratchpad}"
#     ),
# )
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=(
        "You are an assistant with access to tools. Your primary responsibility is to use the ALWAYS USE THE tools if available to solve problems accurately and effectively.\n\n"
        "Guidelines:\n"
        "- NEVER compute answers yourself if a tool is available. Always delegate tasks to tools.\n"
        "Here is a list of tools you can use: query_tool, add_tool, reverse_tool, weather_tool\n"
        "- If a tool is needed:\n"
        "  1. Specify the action using 'Action: [tools]'.\n"
        "  2. Provide input using 'Action Input: [input]'.\n"
        "  3. Wait for the tool's output before proceeding.\n"
        "- If no tool is required, provide the answer directly using 'Final Answer: [answer]'.\n"
        "- DO NOT guess answers or provide explanations if a tool should be used.\n\n"
        "Expected Behavior:\n"
        "- Always think through whether a tool is necessary.\n"
        "- Only compute or provide answers directly if no tool is applicable.\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
    ),
)

# Step 5: Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Step 6: Initialize the Agent with Custom Output Parser
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
    output_parser=DeepSeekOutputParser(),
    allowed_tools=[tool.name for tool in tools]
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Step 7: Use the Agent
if __name__ == "__main__":
    print("Welcome to the LangChain DeepSeek Agent!")
    test_tools()

    while True:
        user_input = input("Ask me anything (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = agent_executor.run(user_input)
            # print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")
