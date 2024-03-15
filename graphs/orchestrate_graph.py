# core libraries
import operator
from typing import Annotated, Sequence, TypedDict

# langchain
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# custom local libraries
from nodes.orchestrate import execute_node, orchestrate_node, memorize_node, plan_node, revise_node, convert_node


# create Orchestrate Agent 
members = ["Plan", "Revise", "Execute", "Memorize", "Convert"]

# construct graph
# The agent state is the input to each node in the graph
class OrchestrateState(TypedDict):
    # The annotation tells the graph that new messages will always be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
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


# define the nodes
workflow = StateGraph(OrchestrateState)
workflow.add_node("Orchestrate", orchestrate_node.node)
workflow.add_node("Plan", plan_node.node)
workflow.add_node("Execute", execute_node.node)
workflow.add_node("Revise", revise_node.node)
workflow.add_node("Memorize", memorize_node.node)
workflow.add_node("Convert", convert_node.node)


# Now connect all the edges in the graph.
for member in members:
    # We want our workers to ALWAYS share their results with the User when done
    workflow.add_edge(member, END)

# conditionally advance from Orchestrate
conditional_map = {k: k for k in members}

# add conditional edges
workflow.add_conditional_edges("Orchestrate", lambda x: x["next"], conditional_map)

# always go to Execute after Convert
workflow.add_edge("Convert", "Execute")

# add entrypoint
workflow.set_entry_point("Orchestrate")

graph = workflow.compile()