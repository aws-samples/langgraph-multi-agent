import os

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.redis import Redis

embeddings = BedrockEmbeddings(model_id='cohere.embed-english-v3')

redis_ip = value = os.getenv('REDIS_IP')
redis_port = value = os.getenv('REDIS_PORT')

rds = Redis.from_existing_index(
    embeddings,
    index_name="plans",
    redis_url=f"redis://{redis_ip}:{redis_port}",
    schema="redis_schema.yaml",
)
    
# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

# main function
def node(state):
    print(f'\n*** Entered Retrieve Node ***\n')
    # collect the User's task from the state
    task = state['messages'][-1].content

    result = rds.similarity_search_with_score(task, k=1)[0]

    # parse redis response    
    document = result[0]
    distance = result[1]
    
    nearest_task = document.page_content
    function_detail = document.metadata['function_detail']
    nearest_plan = document.metadata['plan']
    nearest_code = document.metadata['code']
    
    print(f'Distance to neareast plan: {distance}')

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
            