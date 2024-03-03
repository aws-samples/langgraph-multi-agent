# core libraries
import boto3
from dotenv import load_dotenv, find_dotenv
import json
import os
import pandas as pd

# langchain libraries
from langchain import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Bedrock

# custom local libraries
from vectordb import vectordb

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

# read local .env file
_ = load_dotenv(find_dotenv()) 

# define env vars
access_key_id = os.environ['ACCESS_KEY_ID']
secret_access_key = os.environ['SECRET_ACCESS_KEY']

# define boto clients
bedrock_runtime_client = boto3.client('bedrock-runtime', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

# define language models
model_kwargs = {'temperature': .001, 'max_tokens_to_sample': 10000}
endpoint_url = 'prod.us-west-2.dataplane.bedrock.aws.dev'

llm_modify = Bedrock(model_id='anthropic.claude-instant-v1', client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url=endpoint_url)
llm_formulate = Bedrock(model_id='anthropic.claude-instant-v1', client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url=endpoint_url)
llm_update = Bedrock(model_id='anthropic.claude-v2:1', client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url=endpoint_url)


# read function metadata from disk
with open('state/functions.json', 'r') as file:
    function_dict = json.load(file)

# define string for pybaseball function descriptions that can be used in prompts
pybaseball_functions = '''
- playerid_lookup: Look up a player's various IDs by name.
- statcast: Get pitch-level statcast data for all players and specific dates
- statcast_pitcher: Get pitch-level statcast pitching data for specific pitchers and specific dates.  Use this function to get pitching stats at the game-by-game level.
- statcast_batter: Get pitch-level statcast batting data for specific batters and specific dates.  Use this function to get batting stats at the game-by-game level.
- pitching_stats: Return season-level pitching data.  Use this function to return pitching stats for a season or group of seasons.
- batting_stats: Return season-level batting data.  Use this function to return batting stats for a season or group of seasons.
- schedule_and_record: Return a team's game-level results or future game schedules for a season.
- standings: Get division standings for a given season.
'''


# define helper functions
def create_function_detail_string(functions):
    """
    Convert a list of functions into a string that can be passed into a prompt

    Arguments:
        - functions (list): pybaseball functions used in the plance

    Returns:
        - function_detail_str (str): Formatted string with function metadata
    """

    function_detail_str = ''
    for function in functions:
        function_detail = function_dict[function]
        docs = function_detail['docs']
        data_dictionary = function_detail['data_dictionary']
        function_detail_str += f'Here is the documentation for function {function}:\n'
        function_detail_str += docs
        function_detail_str += f'\nHere is the data dictionary for function {function}:\n'
        function_detail_str += data_dictionary

    return function_detail_str

#   modify
modify_template = '''

Human: Review the execution plan and update it based on the new request while maintaining the original format.

Original Plan:
{existing_plan}

New Request:
{task}

If the original plan already aligns with the new request, return it without any modifications.

Assistant:
'''


def modify_existing_plan(task, existing_plan):
    '''Used to revise the propsed plan based on User feedback'''
    modify_prompt = PromptTemplate(template=modify_template, input_variables=['existing_plan','task'])

    modify_chain = modify_prompt | llm_modify

    result = modify_chain.invoke({'existing_plan':existing_plan, 'task':task})

    return result + '\n\nAre you satisfied with this plan?'

#   formulate
formulate_template = '''

Human: You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. Your expertise is in formulating plans to complete tasks related to baseball data analysis without writing any Python code. Instead, you provide detailed steps that can be executed by an OpenAI Functions Agent within a Python Repl.

When a User presents you with a task, your response will be a structured plan, formatted as a JSON object with the keys "plan" and "functions". The "plan" key will contain a step-by-step guide to complete the task using only the specified pybaseball functions. The "functions" key will list all the pybaseball functions that are utilized in the plan.

Here is the list of pybaseball functions you may use, along with a brief description:

{pybaseball_functions}

Below is an example of a plan for a similar task:

{existing_plan}

Below is an example of how you should format your response:

```
{{
  "plan": "Step 1: Use the `playerid_lookup` function to find the player's IDs.\nStep 2: Retrieve the player's pitching stats using the `pitching_stats` function.\nStep 3: Analyze the retrieved data to determine the player's best season based on ERA and strikeouts.",
  "functions": ["playerid_lookup", "pitching_stats"]
}}
```

Here is the task you need to accomplish:

{task}

RULES:
1. If you need information on a specific player, always use the `playerid_lookup` function to collect the 'key_mlbam' for the player. 

Assistant:
'''

def formulate_new_plan(task, existing_plan):
    '''Used to revise the propsed plan based on User feedback'''
    formulate_new_prompt = PromptTemplate(template=formulate_template, input_variables=['existing_plan','task']).partial(pybaseball_functions=pybaseball_functions)

    formulate_chain = formulate_new_prompt | llm_formulate | JsonOutputParser()

    result = formulate_chain.invoke({'existing_plan':existing_plan, 'task':task})

    return result


#   update
update_template = '''

Human: You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a User create a plan that can be used to complete a task.

Review the task, the original plan, and the details related to the pybaseball functions in the plan.  Then re-write the plan with the following updates:

- be specific about the attributes that should be passed into each pybaseball function
- be specific about which fields should be used in the pybaseball output
- do not attempt to use any fields that are not explicitly mentioned in the data dictionary below

Your response should be a single updated plan that includes changes resulting from the instructions above.  Your plan should NOT include python code,
only the steps necessary to complete the task.

Here is the list of pybaseball functions you may use, along with a brief description:

{pybaseball_functions}

Here is the task:

{task}

Here is the original plan:

{existing_plan}

Here are details about the pybaseball functions in use.  Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.

{function_detail_str}

Assistant: 
'''


def update_plan(task, existing_plan, function_detail_str):
    '''Used to revise the propsed plan based on User feedback'''
    update_prompt = PromptTemplate(template=update_template, input_variables=['existing_plan','task','function_detail_str']).partial(pybaseball_functions=pybaseball_functions)

    update_chain = update_prompt | llm_update

    result = update_chain.invoke({'existing_plan':existing_plan, 'task':task, 'function_detail_str':function_detail_str})

    return result + '\n\nAre you satisfied with this plan?'


# main function
def node(state):
    # collect the User's task from the state
    task = state['messages'][0].content

    # retrieve a collection on plans from vectordb
    plan_collection = vectordb.get_execution_plan_collection()

    # collect the closest plan for the task
    closest_plan = plan_collection.query(query_texts=[task], n_results=1, include=['distances','metadatas','documents'])

    distance = closest_plan['distances'][0][0]
    existing_plan = closest_plan['metadatas'][0][0]['plan']

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
        new_plan = modify_existing_plan(task, existing_plan)

        return {"messages": [HumanMessage(content=new_plan)], 
                "task": task,
                "plan": new_plan,
                "previous_node": "Plan"
            }
    else:
        # no close plan - formulate a one
        print('Formulating a new plan based on User input')
        formulated_plan = formulate_new_plan(task, existing_plan)

        # create a string with function medata
        functions_str = ','.join(formulated_plan['functions'])
        print(f'Collecting metadata for functions {functions_str}')
        function_detail_str = create_function_detail_string(formulated_plan['functions'])

        # update the plan based on this metadata
        print('Modifying plan with function metadata')
        new_plan = update_plan(task, existing_plan, function_detail_str)

        return {"messages": [HumanMessage(content=new_plan)],
                "task": task,
                "plan": new_plan,
                "previous_node": "Plan",
                "function_detail": function_detail_str
            }

