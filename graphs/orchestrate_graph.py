# core libraries
from typing import TypedDict

# langchain
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# custom local libraries
from nodes.orchestrate import (orchestrate_node,
                               retrieve_node,
                               modify_node,
                               execute_graph_node,
                               revise_node,
                               memorize_node,
                               initialize_node,
                               update_node
                                )

# construct graph
# The agent state is the input to each node in the graph
class OrchestrateState(TypedDict):
    # The messages attribute tracks the conversation history
    messages: list[BaseMessage]
    # the session_id keep track of the conversation and is used for Langsmith Threads
    session_id: str
    # The 'previous_node' field indicates what has just completed
    previous_node: str
    # The 'next' field indicates where the workflow should go
    next: str
    # The 'plan' field collects the execution plan
    plan: str
    # The 'task' field collects the task to be executed
    task: str
    # The 'code' field collects the Python code
    code: str
    # The 'function_detail' field collects details on the pybaseball functions in use
    function_detail: str
    # The 'nearest_task' field collect the most semantically similar task
    nearest_task: str
    # The 'nearest_plan' field collect the plan for the most semantically similar task
    nearest_plan: str
    # The 'nearest_code' field collect the code for the most semantically similar task
    nearest_code: str


# define the nodes
workflow = StateGraph(OrchestrateState)
workflow.add_node("Orchestrate", orchestrate_node.node)
workflow.add_node("Retrieve", retrieve_node.node)
workflow.add_node("Modify", modify_node.node)
workflow.add_node("Execute", execute_graph_node.node)
workflow.add_node("Revise", revise_node.node)
workflow.add_node("Memorize", memorize_node.node)
workflow.add_node("Initialize", initialize_node.node)
workflow.add_node("Update", update_node.node)

# add the edges

# conditional edge from Retrieve
retrieve_map = {k: k for k in ['Modify','Initialize']}
workflow.add_conditional_edges("Retrieve", lambda x: x["next"], retrieve_map)

workflow.add_edge("Initialize", "Update")

end_nodes = ['Modify', 'Update', 'Execute', 'Revise', 'Memorize']
for node in end_nodes:
    workflow.add_edge(node, END)

# conditionally advance from Orchestrate
orchestration_nodes = ['Retrieve', 'Revise', 'Execute', 'Memorize']
orchestration_map = {k: k for k in orchestration_nodes}

# add conditional edges
workflow.add_conditional_edges("Orchestrate", lambda x: x["next"], orchestration_map)

# add entrypoint
workflow.set_entry_point("Orchestrate")

graph = workflow.compile()