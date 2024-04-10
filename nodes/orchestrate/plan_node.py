# core libraries
from dotenv import load_dotenv, find_dotenv
import json
import pandas as pd
from typing import List

# langchain libraries
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()

# custom local libraries
from vectordb import vectordb

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

# read local .env file
_ = load_dotenv(find_dotenv()) 

with open('state/functions.json', 'r') as file:
    library_dict = json.load(file)
    
libraries_string = ''
for key in library_dict:
    docs = library_dict[key]['docs']
    libraries_string += f'<{key}>\n{docs}\n</{key}>\n'
    
# Define data models
class ModifiedPlan(BaseModel):
    """Modify a plan based on feedback from the user"""
    plan: str = Field(description="The modified plan after making changes requested by the user.")

class Libraries(BaseModel):
    """Pybaseball libraries used in the plan"""
    library: str = Field(description="pybaseball libraries that are used in the plan.  These will all be imported from the pybaseball library.")
    
class InitialPlan(BaseModel):
    """Initial plan generated to solve the user's task"""
    plan: str = Field(description="The plan that was generated to solve the user's task.")
    libraries: List[Libraries]
    
class UpdatedPlan(BaseModel):
    """Update a plan based on pybaseball library documentation"""
    plan: str = Field(description="The updated plan after making any updates necessary to ensure the correct attributes are passed to each of the pybasell functions.")

# define language models
llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_initial_plan = llm_sonnet.bind_tools([InitialPlan])
llm_modify = llm_sonnet.bind_tools([ModifiedPlan])
llm_update = llm_sonnet.bind_tools([UpdatedPlan])


def collect_library_helpers(libraries):
    '''
    Collect function documentation
    '''
    # extract function names

    libraries_string = ', '.join(libraries)
    print(f'Collecting documentation for {libraries_string}')
    
    helper_string = ''
    for lib in libraries:
        lib_detail = library_dict[lib]
        docs = lib_detail['docs']
        data_dictionary = lib_detail['data_dictionary']
        helper_string += f'Text between the <{lib}_documentation></{lib}_documentation> tags is documentation for the {lib} library.  Consult this section to confirm which attributes to pass into the {lib} library.\n<{lib}_documentation>\n{docs}\n</{lib}_documentation>\n'
        helper_string += f'Text between the <{lib}_dictionary></{lib}_dictionary> tags is the data dictionary for the {lib} library.  Consult this section to confirm which columns are present in the response from the {lib} library.\n<{lib}_dictionary>\n{data_dictionary}\n</{lib}_dictionary>'

    return helper_string


#   modify
MODIFY_SYSTEM_PROMPT = '''
<instructions>Review the original plan and update it based on the new request while maintaining the format.
If the original plan already aligns with the new request, return it without any modifications.
Before updating the plan, do some analysis within <thinking></thinking> tags. 
</instructions>

Text bewteen the <original_task></original_task> tags is the goal of the original plan.
<original_task>
{original_task}
</original_task>

Text bewteen the <original_plan></original_plan> tags is the original plan to be modified.
<original_plan>
{existing_plan}
</original_plan>
'''

def modify_existing_plan(task, nearest_task, nearest_plan, langchain_config): 
    '''Used to revise the propsed plan based on User feedback'''
    modify_prompt = ChatPromptTemplate.from_messages([
        ("system", MODIFY_SYSTEM_PROMPT),
        ("user", "{updates}.  Use ModifiedPlan to describe the plan.") 
    ])

    modify_chain = modify_prompt | llm_modify 

    result = modify_chain.invoke({'existing_plan':nearest_plan, 'original_task': nearest_task, 'updates':task}, config=langchain_config)

    # parse tool response
    modified_plan = [c['input']['plan'] for c in result.content if c['type'] == 'tool_use'][0]
    
    return modified_plan


#   formulate
INITIAL_PLAN_SYSTEM_PROMPT = '''
<instructions>You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your expertise is in formulating plans to complete tasks related to baseball data analysis.  You provide detailed steps that can be executed sequentially to solve the user's task.

Before creating the plan, do some analysis within <thinking></thinking> tags.
</instructions>

Text between the <libraries></libraries> tags is the list of pybaseball libraries you may use, along with their documentation.
<libraries>
{libraries_string}
</libraries>

Text between the <similar_task></similar_task> tags is an example of a similar task to what you are being asked to evaluate.
<similar_task>
{similar_task}
</similar_task>

Text between the <similar_plan></similar_plan> tags is the plan that was executed for the similar task.
<similar_plan>
{existing_plan}
</similar_plan>

Text between the <sample_plan></sample_plan> tags is an example of how you should format your plan.
<sample_plan>
Step 1: Use the `playerid_lookup` function to find the player's IDs.\nStep 2: Retrieve the player's pitching stats using the `pitching_stats` function.\nStep 3: Analyze the retrieved data to determine the player's best season based on ERA and strikeouts.
</sample_plan>

Text between the <rules></rules> tags are rules that must be followed.
<rules> 
1. Include sample Python code to demonstrate how the pybaseball functions should be called where appropriate.
2. Every step that includes a pybaseball function call should include the specific input required for that function call.
3. The last step of the plan should always include a print() statement to describe the results.
</rules>
'''

def formulate_initial_plan(task, existing_plan, similar_task, langchain_config):
    """
    Formulate an initial plan to solve the user's task. 

    Arguments:
        - task (str): task from the user to be solved
        - existing_plan (str): plan associated with the nearest task
        - similar_task (str): nearest plan from the semanitic search
        - langchain_config (dict): configuration for the language model

    Returns:
        - initial_plan (str): Plan generated to solve the task
        - pybaseball_libraries (list): List of pybaseball libraries used in the plan
    """
    
    initial_prompt = ChatPromptTemplate.from_messages([
        ("system", INITIAL_PLAN_SYSTEM_PROMPT),
        ("user", "{task}.  Use InitialPlan to describe the plan."), 
    ])

    initial_plan_chain = initial_prompt | llm_initial_plan 

    result = initial_plan_chain.invoke({'task':task, 'existing_plan':existing_plan, 'similar_task':similar_task, 'libraries_string': libraries_string}, config=langchain_config)
    
    # parse the tool response
    initial_plan = [c['input']['properties']['plan'] for c in result.content if c['type'] == 'tool_use'][0]
    pybaseball_libraries = [c['input']['properties']['libraries'] for c in result.content if c['type'] == 'tool_use'][0]
    pybaseball_libraries = [f['library'] for f in pybaseball_libraries]

    return initial_plan, pybaseball_libraries


#   update
UPDATE_SYSTEM_PROMPT = '''
<instructions>You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to review a plan and ensure that all of the pybaseball functions are being used correctly.

Before updating the plan, do some analysis within <thinking></thinking> tags. Are each of the pybaseball functions being called correctly?
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
    last_message = state['messages'][-1]
    task = last_message.content
    session_id = state['session_id']
    messages = state['messages']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    # retrieve a collection on plans from vectordb
    plan_collection = vectordb.get_execution_plan_collection()

    # collect the closest plan for the task
    closest_plan = plan_collection.query(query_texts=[task], n_results=1, include=['distances','metadatas','documents'])

    distance = closest_plan['distances'][0][0]
    nearest_plan = closest_plan['metadatas'][0][0]['plan']
    nearest_task = closest_plan['documents'][0][0]

    print(f'Distance to neareast plan: {distance}')

    # write task and distance to disk
    task_output_path = 'task_and_distance.csv'
    try:
        task_df = pd.read_csv(task_output_path)
    except:
        task_df = pd.DataFrame(columns=['task', 'distance'])

    task_df.loc[len(task_df.index)] = [task, distance]
    task_df.to_csv(task_output_path, index=False)

    # determine path based on whether we have a similar "enough" plan
    if distance <= threshold:
        # close enough plan - modify it with our current task
        print('Modifying nearest plan with User input')
        new_plan = modify_existing_plan(task, nearest_task, nearest_plan, langchain_config)
        
        messages.append(AIMessage(content=new_plan))

        return {"messages": messages, 
                "task": task,
                "plan": new_plan,
                "previous_node": "Plan"
            }
    else:
        # no close plan - formulate a one
        print('Formulating a new plan based on User input')
        
        # forumalte an initial plan
        initial_plan, pybaseball_libraries = formulate_initial_plan(task, nearest_plan, nearest_task, langchain_config)
        
        # collect documentation on functions
        helper_string = collect_library_helpers(pybaseball_libraries)
        
        # update plan based on helper string detail
        print('Updating plan based function documentation')
        updated_plan = update_plan(task, initial_plan, helper_string, langchain_config)
        
        messages.append(AIMessage(content=updated_plan))

        return {"messages": messages,
                "task": task,
                "plan": updated_plan,
                "previous_node": "Plan",
                "function_detail": helper_string
            }
            