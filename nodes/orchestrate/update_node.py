# core libraries
from dotenv import load_dotenv, find_dotenv
import json

# langchain libraries
from langchain_anthropic import ChatAnthropic
#from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

with open('state/functions.json', 'r') as file:
    library_dict = json.load(file)


class UpdatedPlan(BaseModel):
    """Update a plan based on pybaseball library documentation"""
    plan: str = Field(description="The updated plan after making any updates necessary to ensure the correct attributes are passed to each of the pybasell functions.")

# define language models
#llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
#llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_update = llm_opus.bind_tools([UpdatedPlan])

#   update
UPDATE_SYSTEM_PROMPT = '''
<instructions>You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to review a plan and ensure that all of the pybaseball functions are being used correctly.

Before updating the plan, do some analysis within <thinking></thinking> tags. Are the correct attributes with the correct data types being passed to each call to a pybaseball library?
</instructions>

Text between the <task></task> tags is the goal of the plan.
<task>
{task}
</task>

Text between the <current_plan></current_plan> tags is the current plan to be modified.
<current_plan>
{current_plan}
</current_plan>

Text bewteen the <function_detail></function_detail> tags documentation on the pybaseball functions in use
<function_detail>
{function_detail}
</function_detail>

Make any updates necessary to ensure the correct attributes are being passed to each of the pybaseball functions.
'''


def update_plan(task, current_plan, function_detail, langchain_config):
    '''Used to revise the propsed plan based on function documentation'''
    
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", UPDATE_SYSTEM_PROMPT),
        ("user", "Review the current plan and make any updates necessary to ensure the correct attributes are being passed to the pybaseball functinons.  Use UpdatedPlan to describe the plan."), 
    ])

    update_chain = update_prompt | llm_update 
    
    result = update_chain.invoke({'task':task, 'current_plan':current_plan, 'function_detail':function_detail}, config=langchain_config)
    
    # parse tool response
    tool_calls = result.tool_calls
    updated_plan = [t['args']['plan'] for t in tool_calls if t['name'] == 'UpdatedPlan'][0]
    
    return updated_plan + '\n\nAre you satisfied with this plan?'


# main function
def node(state):
    print(f'\n*** Entered Update Node ***\n')
    # collect the User's task from the state
    task = state['task']
    session_id = state['session_id']
    messages = state['messages']
    plan = state['plan']
    function_detail = state['function_detail']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
        
    # update plan based on helper string detail
    print('Updating plan based function documentation')
    updated_plan = update_plan(task, plan, function_detail, langchain_config)
    
    messages.append(AIMessage(content=updated_plan))
    
    # update state
    state['messages'] = messages
    state['plan' ] = updated_plan
    state['previous_node'] = 'Update'
    
    return state
            