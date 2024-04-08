# core libraries
import operator
from typing import Annotated, Sequence, TypedDict

# langchain
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

# custom libraries
from nodes.execute import generate_node, execute_node

class ExecuteState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    # The 'messages' attribute keeps track of the conversation history for the execution
    messages: list
    # the 'session_id' keeps track of the conversation and is used for Langsmith Threads
    session_id: str
    # The 'plan' field collects the execution plan
    plan: str
    # The 'task' field collects the task to be executed
    task: str
    # The 'function_detail' field collects details on the pybaseball functions in use
    function_detail: str
    # The 'iterations' field collects the number of code iterations
    iterations: int
    # The 'result' field collects the result of executing the python code
    result: str
    # The 'final_result' field collects the final result of the execution plan
    final_result: str
    # The 'successful_code' field collects the successfully executed code
    successful_code: list
    
    
### Edges
def decide_to_finish(state):
    """
    Determines whether to finish (re-try code 3 times)

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    #iter = state["iterations"]
    if state['final_result']:
        return "end"
    else:
        return "generate"
    
workflow = StateGraph(ExecuteState)

# Define the nodes
workflow.add_node("generate", generate_node.node)  # generation solution
workflow.add_node("execute", execute_node.node)  # check execution

# Build graph
workflow.set_entry_point("generate")

# add edges
workflow.add_edge("generate", "execute")

# add conditional edges
workflow.add_conditional_edges(
    "execute",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

# Compile
graph = workflow.compile()