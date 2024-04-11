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

Before updating the plan, do some analysis within <thinking></thinking> tags. Are the correct attributes being passed to each call to a pybaseball library?
</instructions>

Text between the <task></task> tags is the goal of the plan.
<task>
{task}
</task>

Text between the <current_plan></current_plan> tags is the current plan to be modified.
<current_plan>
{current_plan}
</current_plan>

Text bewteen the <helper_string></helper_string> tags documentation on the pybaseball functions in use
<helper_string>
{helper_string}
</helper_string>

Make any updates necessary to ensure the correct attributes are being passed to each of the pybaseball functions.
'''


def collect_library_helpers(libraries):
    '''
    Collect pybaseball library documentation
    '''
    
    print(f'Collecting documentation for {libraries}')
    lib_list =[i.strip() for i in libraries.split(',')]
    
    helper_string = ''
    for lib in lib_list:
        lib_detail = library_dict[lib]
        docs = lib_detail['docs']
        #data_dictionary = lib_detail['data_dictionary']
        helper_string += f'Text between the <{lib}_documentation></{lib}_documentation> tags is documentation for the {lib} library.  Consult this section to confirm which attributes to pass into the {lib} library.\n<{lib}_documentation>\n{docs}\n</{lib}_documentation>\n'
        #helper_string += f'Text between the <{lib}_dictionary></{lib}_dictionary> tags is the data dictionary for the {lib} library.  Consult this section to confirm which columns are present in the response from the {lib} library.\n<{lib}_dictionary>\n{data_dictionary}\n</{lib}_dictionary>'

    return helper_string


def update_plan(task, current_plan, helper_string, langchain_config):
    '''Used to revise the propsed plan based on function documentation'''
    
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", UPDATE_SYSTEM_PROMPT),
        ("user", "Review the current plan and make any updates necessary to ensure the correct attributes are being passed to the pybaseball functinons.  Use UpdatedPlan to describe the plan."), 
    ])

    update_chain = update_prompt | llm_update 
    
    result = update_chain.invoke({'task':task, 'current_plan':current_plan, 'helper_string':helper_string}, config=langchain_config)
    
    # parse tool response
    updated_plan = [c['input']['plan'] for c in result.content if c['type'] == 'tool_use'][0]
    
    return updated_plan + '\n\nAre you satisfied with this plan?'


# main function
def node(state):
    # collect the User's task from the state
    task = state['task']
    session_id = state['session_id']
    messages = state['messages']
    helper_string = state['function_detail']
    plan = state['plan']
    pybaseball_libraries = state['pybaseball_libraries']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    # collect documentation on functions
    helper_string = collect_library_helpers(pybaseball_libraries)
        
    # update plan based on helper string detail
    print('Updating plan based function documentation')
    updated_plan = update_plan(task, plan, helper_string, langchain_config)
    
    messages.append(AIMessage(content=updated_plan))

    return {"messages": messages,
            "task": task,
            "plan": updated_plan,
            "previous_node": "Update"
        }
            