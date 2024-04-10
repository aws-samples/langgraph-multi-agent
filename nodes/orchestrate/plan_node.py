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
    function_dict = json.load(file)
    
functions_string = ''
for key in function_dict:
    docs = function_dict[key]['docs']
    functions_string += f'<{key}>\n{docs}\n</{key}>\n'
    
# Define data models
class ModifiedPlan(BaseModel):
    """Modify a plan"""
    plan: str = Field(description="The modified plan after making changes requested by the user.")
    
class Functions(BaseModel):
    """Pybaseball libraries"""
    function: str = Field(description="pybaseball functions that are used in the plan")
    
class InitialPlan(BaseModel):
    """Initial plan generated to solve the user's task"""
    plan: str = Field(description="The plan that was generated to solve the user's task.")
    functions: List[Functions]
    
class UpdatedPlan(BaseModel):
    """Update a plan"""
    plan: str = Field(description="The updated plan after making any updates necessary to ensure the correct attributes are passed to each of the pybasell functions.")

# define language models
llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_initial_plan = llm_sonnet.with_structured_output(InitialPlan, include_raw=False)
llm_modify = llm_haiku.with_structured_output(ModifiedPlan, include_raw=False)
llm_update = llm_sonnet.with_structured_output(UpdatedPlan, include_raw=False)

def collect_function_helpers(functions):
    '''
    Collect function documentation
    '''
    # extract function names
    function_list = [f.function for f in functions]
    functions_string = ', '.join(function_list)
    print(f'Collecting documentation for {functions_string}')
    
    helper_string = ''
    for function in function_list:
        function_detail = function_dict[function]
        docs = function_detail['docs']
        helper_string += f'Text between the <{function}_documentation></{function}_documentation> tags is documentation for the {function} function.  Consult this section to confirm which attributes to pass into the {function} function.\n<{function}_documentation>\n{docs}\n</{function}_documentation>\n'

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
        ("user", "{updates}") 
    ])

    modify_chain = modify_prompt | llm_modify 

    result = modify_chain.invoke({'existing_plan':nearest_plan, 'original_task': nearest_task, 'updates':task}, config=langchain_config)

    return result.plan + '\n\nAre you satisfied with this plan?'


#   formulate
INITIAL_PLAN_SYSTEM_PROMPT = '''
<instructions>You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your expertise is in formulating plans to complete tasks related to baseball data analysis.  You provide detailed steps that can be executed sequentially to solve the user's task.

Before creating the plan, do some analysis within <thinking></thinking> tags.
</instructions>

Text between the <functions></functions> tags is the list of pybaseball functions you may use, along with their documentation.
<functions>
{functions_string}
</functions>

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
    '''Used to revise the propsed plan based on User feedback'''
    initial_prompt = ChatPromptTemplate.from_messages([
        ("system", INITIAL_PLAN_SYSTEM_PROMPT),
        ("user", "{task}"), 
    ])

    initial_plan_chain = initial_prompt | llm_initial_plan 

    result = initial_plan_chain.invoke({'task':task, 'existing_plan':existing_plan, 'similar_task':similar_task, 'functions_string': functions_string}, config=langchain_config)

    return result


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
        ("user", "Review the current plan and make any updates necessary to ensure the correct attributes are being passed to the pybaseball functinons."), 
    ])

    update_chain = update_prompt | llm_update 
    
    result = update_chain.invoke({'task':task, 'current_plan':current_plan, 'helper_string':helper_string}, config=langchain_config)
    
    return result.plan + '\n\nAre you satisfied with this plan?'


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
        initial_plan = formulate_initial_plan(task, nearest_plan, nearest_task, langchain_config)
        
        # collect documentation on functions
        helper_string = collect_function_helpers(initial_plan.functions)
        
        # update plan based on helper string detail
        print('Updating plan based function documentation')
        updated_plan = update_plan(task, initial_plan.plan, helper_string, langchain_config)
        
        messages.append(AIMessage(content=updated_plan))

        return {"messages": messages,
                "task": task,
                "plan": updated_plan,
                "previous_node": "Plan",
                "function_detail": helper_string
            }
            