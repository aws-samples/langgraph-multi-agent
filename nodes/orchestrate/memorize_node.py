import os

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain_core.messages import AIMessage

embeddings = BedrockEmbeddings(model_id='cohere.embed-english-v3')

redis_ip = value = os.getenv('REDIS_IP')
redis_port = value = os.getenv('REDIS_PORT')

redis_url = f"redis://{redis_ip}:{redis_port}"

rds = Redis.from_existing_index(
    embeddings,
    index_name="plans",
    redis_url=redis_url,
    schema="redis_schema.yaml",
)

def node(state):
    '''Used to submit a successful execution to long term memory'''
    print('\n*** Entered Memorize Node ***\n')
    # collect task and plan from state
    task = state['task']
    updated_plan = state['plan']
    code = state['code']
    function_detail = state['function_detail']
    messages = state['messages']
    
    # search for nearest task
    result = rds.similarity_search_with_score(task, k=1)[0]
    
    # parse redis response    
    distance = result[1]

    metadata = {
            "plan": updated_plan,
            "code": code,
            "function_detail": function_detail
            }

    if distance <= .1:
        # remove existing document so that it can be replaced
        id = result[0].metadata['id']
        rds.delete(ids=[id], redis_url=redis_url)
        rds.add_texts([task], [metadata])
        response = 'Thank you, task has been updated in memory'
    else:
        rds.add_texts([task], [metadata])
        response = 'Thank you, task has been commited to memory'

    
    messages.append(AIMessage(content=response))
    
    # update state
    state['messages'] = messages

    return state