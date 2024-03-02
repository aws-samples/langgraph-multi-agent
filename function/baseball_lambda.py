# core libraries
import pickle

# langchain
from langchain_core.messages import HumanMessage

# custom local libraries
from graphs import orchestrate_graph

# read function metadata from disk
state_dict_path = 'state/state.pkl'
    
def execute_workflow(task, session_id):
    # convert input to HumanMessage
    human_message = HumanMessage(content=task, name='User')
    
    # load previous state
    try:
        with open(state_dict_path, 'rb') as file:
            state_dict = pickle.load(file)
    except:
        state_dict = {}
    
    # check for previous state
    if session_id in state_dict:
        state_dict[session_id]['messages'].append(human_message)
    else:
        state_dict[session_id] = {}
        state_dict[session_id]['messages'] = [human_message]
        
    # execute
    for s in orchestrate_graph.graph.stream(state_dict[session_id]):
        if "__end__" not in s:
            print(s)
            print("----")
            
    # update state
    state_dict[session_id] = s['__end__'] 
    
    # write to disk
    with open(state_dict_path, 'wb') as file:
        pickle.dump(state_dict, file)
    
    # collect response
    last_message = s['__end__']['messages'][-1].content
    
    return last_message