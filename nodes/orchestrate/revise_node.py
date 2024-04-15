#from langchain_community.chat_models import BedrockChat
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.base import RunnableParallel


# Define data models
class RevisedPlan(BaseModel):
    """Use this tool to describe the plan created to solve a user's task."""
    plan: str = Field(description="The revised plan after making changes requested by the user.")

class RevisedTask(BaseModel):
    """Use this tool to describe the task after updates from the user"""
    task: str = Field(description="The revised task after making changes requested by the user.  If there are no changes to the task, this will be the same as the original task")

# define language models
llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_revise = llm_opus
llm_formatted_plan = llm_haiku.bind_tools([RevisedPlan])
llm_formatted_task = llm_haiku.bind_tools([RevisedTask])

REVISE_SYSTEM_PROMPT = '''
<instructions>You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a user create a plan that can be used to complete a task.

Review the original plan and the details related to the pybaseball functions in the plan.  Then revise the plan based on feedback from the user.
</instructions>

Text between the <task></task> tags is the original task of the plan to be revised.
<task>
{task}
</task>

Text between the <original_plan></original_plan> tags is the original plan to be revised.
<original_plan>
{plan}
</original_plan>

Text bewteen the <function_detail></function_detail> tags is information about the pybaseball functions in use.  
<function_detail> 
{function_detail}
<function_detail>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.
2. Every step that includes a pybaseball function call should include the specific input required for that function call
</rules>
'''

revise_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REVISE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"), 
])

format_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Use the RevisedPlan tool to describe the plan.'),
    ("user", "{plan}"), 
])

task_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Use the RevisedTask tool to describe the any necssary updates to the original task.\n\n<original_task>{task}</original_task>'),
    ("user", "{revision}"), 
])

# create individual chain to revise the plan and the task
plan_chain = revise_prompt_template | llm_revise | format_prompt | llm_formatted_plan
task_chain = task_prompt | llm_formatted_task

# combine into a parallel chain
parallel_formatting_chain = RunnableParallel(plan=plan_chain, task=task_chain)


def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    print(f'\n*** Entered Revise Node ***\n')
    
    # collect metadata from state
    plan = state['plan']
    function_detail = state['function_detail']
    messages = state['messages']
    session_id = state['session_id']
    task = state['task']
    
    revision = state['messages'][-1].content
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    # invoke revise chain
    result = parallel_formatting_chain.invoke({'plan': plan, 'revision':revision, 'function_detail': function_detail, 'task': task, 'messages': messages}, config=langchain_config)
    
    # parse the tool response
    plan_tools = result['plan'].tool_calls
    revised_plan = [t['args']['plan'] for t in plan_tools if t['name'] == 'RevisedPlan'][0]

    task_tools = result['task'].tool_calls
    task = [t['args']['task'] for t in task_tools if t['name'] == 'RevisedTask'][0]
    
    revised_plan += '\n\nAre you satisfied with this plan?'
    
    messages.append(AIMessage(content=revised_plan))
    
    # update state
    state['plan'] = revised_plan
    state['previous_node'] = 'Revise'
    state['task'] = task
    state['messages'] = messages
    
    return state