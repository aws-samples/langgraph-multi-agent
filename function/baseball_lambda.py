# core libraries
import os
import pickle
import redis

# langchain
from langchain_core.messages import HumanMessage

# custom local libraries
from graphs import orchestrate_graph

redis_ip = value = os.getenv('REDIS_IP')
redis_port = value = os.getenv('REDIS_PORT')

# establish redis connection
r = redis.Redis(host=redis_ip, port=redis_port, db=0)

def execute_workflow(task, session_id):
    # convert input to HumanMessage
    human_message = HumanMessage(content=task, name='User')
    
    # load previous state
    state_dict = r.get(session_id)
    if state_dict:
        state_dict = pickle.loads(state_dict)
        state_dict['messages'].append(human_message)
    else:
        state_dict = {}
        state_dict['session_id'] = session_id
        state_dict['messages'] = [human_message]
        
    # execute
    for s in orchestrate_graph.graph.stream(state_dict):
        for key, value in s.items():
            pass
            
    # update state
    state_dict = s[key] 

    # write to redis
    pickle_state = pickle.dumps(state_dict)
    r.set(session_id, pickle_state)

    # collect response
    last_message = s[key]['messages'][-1].content
    
    return last_message