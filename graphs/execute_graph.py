import re
from typing import Dict, TypedDict

from langgraph.graph import END, StateGraph

from nodes.execute import generate_node, execute_node

class ExecuteState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    # The 'code' field collects the Python code
    code: str
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
    # The 'successful_code' field collects the code that has been executed successfully so-far
    successful_code :str
    
    
### Edges
def decide_to_finish(state):
    """
    Determines whether to finish (re-try code 3 times.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    iter = state["iterations"]

    if 'error' not in state['result'].lower() or iter == 3:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "end"
    else:
        # We have relevant documents, so generate answer
        return "generate"
    
workflow = StateGraph(ExecuteState)

# Define the nodes
workflow.add_node("generate", generate_node.node)  # generation solution
workflow.add_node("execute", execute_node.node)  # check execution

# Build graph
workflow.set_entry_point("execute")

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