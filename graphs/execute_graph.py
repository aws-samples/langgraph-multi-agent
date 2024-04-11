# core libraries
from typing import TypedDict

# langchain
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

# custom libraries
from nodes.execute import generate_node, execute_node, summarize_node

class ExecuteState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    # The 'messages' attribute keeps track of the conversation history for the execution
    messages: list[BaseMessage]
    # the 'session_id' keeps track of the conversation and is used for Langsmith Threads
    session_id: str
    # The 'plan' field collects the execution plan
    plan: str
    # The 'task' field collects the task to be executed
    task: str
    # The 'function_detail' field collects details on the pybaseball functions in use
    function_detail: str
    # The 'successful_code' field collects the successfully executed code
    successful_code: list
    # The 'nearest_code' field collect the code for the most semantically similar task
    nearest_code: str
    
    
### Edges
def decide_to_finish(state):
    """
    Determines whether to finish (re-try code 3 times)

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    # Determine whether there is a tool use call
    if state['messages'][-1].response_metadata['stop_reason'] == 'tool_use':
        return "execute"
    else:
        return "summarize"
    
workflow = StateGraph(ExecuteState)

# Define the nodes
workflow.add_node("generate", generate_node.node)  # generation solution
workflow.add_node("execute", execute_node.node)  # executed code
workflow.add_node("summarize", summarize_node.node)  # executed code

# Build graph
workflow.set_entry_point("generate")

# add edges
workflow.add_edge("execute", "generate")
workflow.add_edge("summarize", END)

# add conditional edges
workflow.add_conditional_edges(
    "generate",
    decide_to_finish,
    {
        "execute": "execute",
        "summarize": "summarize"
    },
)

# Compile
graph = workflow.compile()