# core libraries
from dotenv import load_dotenv, find_dotenv
import json
import pandas as pd

# langchain libraries
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

# custom local libraries
from vectordb import vectordb

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

# read local .env file
_ = load_dotenv(find_dotenv()) 

# define language models
sonnet_model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
haiku_model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
model_kwargs={'temperature': 0, "max_tokens": 10000}

llm_modify = BedrockChat(model_id=sonnet_model_id, model_kwargs=model_kwargs)
llm_formulate = BedrockChat(model_id=sonnet_model_id, model_kwargs=model_kwargs)
llm_update = BedrockChat(model_id=sonnet_model_id, model_kwargs=model_kwargs)

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

"""
pybaseball_functions = '''
- playerid_lookup: Look up a player's various IDs by name.
- statcast: Get pitch-level statcast data for all players and specific dates
- schedule_and_record: Return a team's game-level results or future game schedules for a season.
'''
"""
def extract_plan(message):
    '''
    Helper function to extract the final plan
    '''
    text = message.content
    start_tag = '<plan>'
    end_tag = '</plan>'
    start_index = text.find(start_tag)
    end_index = text.find(end_tag, start_index + len(start_tag))
    if start_index != -1 and end_index != -1:
        return text[start_index + len(start_tag):end_index]
    else:
        return None  # Return None if tags are not found


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
        function_detail_str += f'Text between the <{function}_documentation></{function}_documentation> tags is documentation for the {function} function.  Consult this section to confirm which attributes to pass into the {function} function.\n<{function}_documentation>\n{docs}\n</{function}_documentation>\n'
        function_detail_str += f'Text between the <{function}_dictionary></{function}_dictionary> tags is the data dictionary for the {function} function.\n<{function}_dictionary>\n{data_dictionary}\n</{function}_dictionary>'

    return function_detail_str

#   modify
MODIFY_SYSTEM_PROMPT = '''
Review the original plan and update it based on the new request while maintaining the format.

If the original plan already aligns with the new request, return it without any modifications.

Text bewteen the <original_plan></original_plan> tags is the original plan to be modified.
<original_plan>
{existing_plan}
</original_plan>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. Always return the plan between <plan></plan> tags 
</rules>
'''


def modify_existing_plan(task, existing_plan, langchain_config):
    '''Used to revise the propsed plan based on User feedback'''
    modify_prompt = ChatPromptTemplate.from_messages([
        ("system", MODIFY_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ])

    modify_chain = modify_prompt | llm_modify | RunnableLambda(extract_plan)

    result = modify_chain.invoke({'existing_plan':existing_plan, 'messages':[HumanMessage(content=task)]}, config=langchain_config)

    return result + '\n\nAre you satisfied with this plan?'

#   formulate
FORMULATE_SYSTEM_PROMPT = '''
You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. Your expertise is in formulating plans to complete tasks related to baseball data analysis without writing any Python code. Instead, you provide detailed steps that can be executed by an OpenAI Functions Agent within a Python Repl.

When a User presents you with a task, your response will be a structured plan, formatted as a JSON object with the keys "plan" and "functions". The "plan" key will contain a step-by-step guide to complete the task using only the specified pybaseball functions. The "functions" key will list all the pybaseball functions that are utilized in the plan.

Text between the <functions></functions> tags is the list of pybaseball functions you may use, along with a brief description.
<functions>
{pybaseball_functions}
</functions>

Text between the <similar_plan></similar_plan> tags is an example of a plan for a similar task.
<similar_plan>
{existing_plan}
</similar_plan>

Text between the <sample_response></sample_response> tags is an example of how you should format your respoonse.
<sample_response>
```json
{{
  "plan": "Step 1: Use the `playerid_lookup` function to find the player's IDs.\nStep 2: Retrieve the player's pitching stats using the `pitching_stats` function.\nStep 3: Analyze the retrieved data to determine the player's best season based on ERA and strikeouts.",
  "functions": ["playerid_lookup", "pitching_stats"]
}}
```
</sample_response>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. If you need information on a specific player, always use the `playerid_lookup` function to collect the 'key_mlbam' for the player. 
2. Include sample Python code to demonstrate how the pybaseball functions should be called where appropriate
3. Every step that includes a pybaseball function call should include the specific input required for that function call
</rules>
'''

def formulate_new_plan(task, existing_plan, langchain_config):
    '''Used to revise the propsed plan based on User feedback'''
    formulate_new_prompt = ChatPromptTemplate.from_messages([
        ("system", FORMULATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ]).partial(pybaseball_functions=pybaseball_functions, existing_plan=existing_plan)

    formulate_chain = formulate_new_prompt | llm_formulate | JsonOutputParser()

    result = formulate_chain.invoke({'messages':[HumanMessage(content=task)]}, config=langchain_config)

    return result


#   update
UPDATE_SYSTEM_PROMPT = '''
You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a User create a plan that can be used to complete a task.

Review the original plan and the details related to the pybaseball functions in the plan.  Then update the plan with the following updates:

- be specific about the attributes that should be passed into each pybaseball function.  Review the documentation between the <function_detail></function_detail> tags below to confirm that the correct attributes are being passed into each function. 
- be specific about which fields should be used in the pybaseball output
- do not attempt to use any fields that are not explicitly mentioned in the data dictionary below

Your response should be a single updated plan that includes changes resulting from the instructions above.
Before updating the plan, do some analysis within <thinking></thinking> tags. Check the documentation for the functions to ensure the correct attributes are being passed and the correct output is expected.


Text between the <original_plan></original_plan> tags is the original plan to be modified.
<original_plan>
{existing_plan}
</original_plan>

Text between the <function_detail></function_detail> tags is documentation on the functions in use.
<function_detail> 
{function_detail_str}
</function_detail>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. If you need information on a specific player, always use the `playerid_lookup` function to collect the 'key_mlbam' for the player. 
2. Include sample Python code to demonstrate how the pybaseball functions should be called where appropriate
3. Always return the plan between <plan></plan> tags
4. Every step that includes a pybaseball function call should include the specific input required for that function call.  Double check the documentation between the <function_detail></function_detail> tags to be sure you are passing the correct attributes.
5. Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.
</rules>
'''


def update_plan(task, existing_plan, function_detail_str, langchain_config):
    '''Used to revise the propsed plan based on User feedback'''
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", UPDATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ]).partial(existing_plan=existing_plan, function_detail_str=function_detail_str)

    update_chain = update_prompt | llm_update | RunnableLambda(extract_plan)

    result = update_chain.invoke({'messages':[HumanMessage(content=task)]}, config=langchain_config)
    
    print('UPDATE RESULT:')
    print(result)

    return result + '\n\nAre you satisfied with this plan?'


# main function
def node(state):
    # collect the User's task from the state
    last_message = state['messages'][-1]
    task = last_message.content
    session_id = state['session_id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

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
        new_plan = modify_existing_plan(task, existing_plan, langchain_config)

        return {"messages": [HumanMessage(content=new_plan)], 
                "task": task,
                "plan": new_plan,
                "previous_node": "Plan"
            }
    else:
        # no close plan - formulate a one
        print('Formulating a new plan based on User input')
        formulated_plan = formulate_new_plan(task, existing_plan, langchain_config)

        # create a string with function medata
        functions_str = ','.join(formulated_plan['functions'])
        print(f'Collecting metadata for functions {functions_str}')
        function_detail_str = create_function_detail_string(formulated_plan['functions'])

        # update the plan based on this metadata
        print('Modifying plan with function metadata')
        new_plan = update_plan(task, existing_plan, function_detail_str, langchain_config)

        return {"messages": [HumanMessage(content=new_plan)],
                "task": task,
                "plan": new_plan,
                "previous_node": "Plan",
                "function_detail": function_detail_str
            }

