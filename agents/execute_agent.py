from dotenv import load_dotenv, find_dotenv
import json
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# define tools
tools = [PythonREPLTool()]
tool_executor = ToolExecutor(tools)

# define a model
#model = ChatOpenAI(temperature=0, streaming=True, model='gpt-4-turbo-preview')
model = ChatOpenAI(temperature=0, streaming=True, model='gpt-4')

# define the output format
class ExecutorResponse(BaseModel):
    """Final response to the user from the Executor node"""
    answer: str = Field(description="The answer to the question posed by User.")
    python_code: str = Field(description="All Python code that was successfully executed.")
    modifications: str = Field(description="A descriptions of the modification that needed to be made to the plan during execution.  If there were no modifications required, respond with 'none'")
    plan: str = Field(description="The exact plan that was executed after any modifications (if any) were made.  Do not summarize the plan.  Record the plan exactly as it was executed.")

# bind to model
functions = [convert_to_openai_function(t) for t in tools]
functions.append(convert_to_openai_function(ExecutorResponse))
model = model.bind_functions(functions)

# define Agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we need to check what type of function call it is
    elif last_message.additional_kwargs["function_call"]["name"] == "ExecutorResponse":
        return "end"
    # Otherwise we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # format input
    query = action.tool_input['__arg1']
    action.tool_input = {'query': query}
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
workflow.add_edge('action', 'agent')

# compile
graph = workflow.compile()
