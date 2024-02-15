# core libraries
import operator
from typing import Annotated, Sequence, TypedDict

# langchain
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# custom local libraries
from nodes import execute_node, memorize_node, plan_node, revise_node

# define language model
#llm = ChatOpenAI(model="gpt-3.5-turbo")
#llm = ChatOpenAI(temperature=0, streaming=True, model='gpt-4-turbo-preview')
llm = ChatOpenAI(temperature=0, streaming=True, model='gpt-4')

# create Orchestrate Agent 
members = ["Plan", "Revise", "Execute", "Memorize"]
system_prompt = (
'''
You are a supervisor managing a conversation between workers: {members} and a human User. Follow these steps to assist the User with their task:

1. Use "Plan" to create an initial plan to solve the User's task. Only use "Plan" once per conversation.
2. Use "Revise" to make revisions to a plan based on feedback from the User. Always use "Revise" to make updates and revisions after you have used "Plan" to create the initial plan.
3. Use "Execute" to carry out the plan once approved by the User.
4. Use "Memorize" to commit a task to memory when the User is satisfied with the result.

Given the User's request below, respond with the worker to act next.
'''
)

# Our team supervisor is an LLM node. It just picks the next agent to process
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next worker.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": members},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Select one of: {members}",
        ),
    ]
).partial(members=", ".join(members))


orchestrate_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# construct graph
# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


# define the nodes
workflow = StateGraph(AgentState)
workflow.add_node("Plan", plan_node.node)
workflow.add_node("Execute", execute_node.node)
workflow.add_node("Orchestrate", orchestrate_chain)
workflow.add_node("Revise", revise_node.node)
workflow.add_node("Memorize", memorize_node.node)


# Now connect all the edges in the graph.
for member in members:
    # We want our workers to ALWAYS share their results with the User when done
    workflow.add_edge(member, END)

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}

workflow.add_conditional_edges("Orchestrate", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("Orchestrate")

graph = workflow.compile()