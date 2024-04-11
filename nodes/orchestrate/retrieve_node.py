# core libraries
import pandas as pd
from typing import List

# langchain libraries
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# custom local libraries
from vectordb import vectordb

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

# main function
def node(state):
    # collect the User's task from the state
    task = state['messages'][-1].content

    # retrieve a collection on plans from vectordb
    plan_collection = vectordb.get_execution_plan_collection()

    # collect the closest plan for the task
    closest_plan = plan_collection.query(query_texts=[task], n_results=1, include=['distances','metadatas','documents'])

    distance = closest_plan['distances'][0][0]
    nearest_plan = closest_plan['metadatas'][0][0]['plan']
    nearest_code = closest_plan['metadatas'][0][0]['code']
    function_detail = closest_plan['metadatas'][0][0]['function_detail']
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
        next_step = 'Modify'
        known_plan = True
    else:
        next_step = 'Initialize'
        known_plan = False
        
    # update state
    state['next'] = next_step
    state['nearest_plan'] = nearest_plan
    state['nearest_task'] = nearest_task
    state['nearest_code'] = nearest_code
    state['function_detail'] = function_detail
    state['task'] = task
    state['previous_node'] = 'Retrieve'
    state['known_plan'] = known_plan
    
    return state
            