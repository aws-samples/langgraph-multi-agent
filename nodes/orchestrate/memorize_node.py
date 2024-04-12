import chromadb
import pandas as pd

from langchain_core.messages import AIMessage


def commit_to_memory(task: str, updated_plan: str, code: str, function_detail: str) -> dict:
    "Use this tool when you need to commit a task to memory"

    execution_plan_path = 'vectordb/execution_plan.csv'

    steps_df = pd.read_csv(execution_plan_path)
    # collect distance
    chroma_client = chromadb.Client()
    plan_collection = chroma_client.get_collection(name='execution_plan')
    result = plan_collection.query(query_texts=task, n_results=1)
    distance = result['distances'][0][0]

    if distance > .01: # add new task
        steps_df.loc[len(steps_df.index)] = [task, updated_plan, code, function_detail]
        response = 'Thank you, task has been commited to memory'
    else: # update existing task
        same_task = result['documents'][0][0]
        same_task_index = steps_df.index[steps_df['task'] == same_task].tolist()[0]
        steps_df.loc[same_task_index] = [task, updated_plan, code, function_detail]
        response = 'Thank you, task has been updated in memory'

    steps_df.to_csv(execution_plan_path, index=False)

    return response


def node(state):
    '''Used to submit a successful execution to long term memory'''
    print(f'\n*** Entered Memorize Node ***\n')
    # collect task and plan from state
    task = state['task']
    updated_plan = state['plan']
    code = state['code']
    function_detail = state['function_detail']
    messages = state['messages']

    # commit
    response = commit_to_memory(task=task, updated_plan=updated_plan, code=code, function_detail=function_detail)
    
    messages.append(AIMessage(content=response))
    
    # update state
    state['messages'] = messages

    return state