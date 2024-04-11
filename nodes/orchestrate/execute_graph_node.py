from graphs import execute_graph
from langchain_core.messages import AIMessage


def node(state):
    # collect metadata from state
    plan = state['plan']
    task = state['task']
    function_detail = state['function_detail']
    session_id = state['session_id']
    nearest_code = state['nearest_code']
    messages = state['messages']
    known_plan = state['known_plan']

    inputs = {"plan": plan, 
              "task": task, 
              "function_detail": function_detail, 
              "session_id": session_id, 
              'messages':[], 
              'successful_code': [], 
              'known_plan': known_plan,
              'nearest_code': nearest_code}
    
        # define model
    if known_plan:
        print('Known plan. Executing with Sonnet')
    else:
        print('Unknown plan. Executing with Opus')

    for s in execute_graph.graph.stream(inputs, {"recursion_limit": 100}):
        for key, value in s.items():
            if key == 'summarize':
                successful_code = s[key]['successful_code']
                final_result = s[key]['messages'][-1].content
                
                successful_code_string = '\n'.join(successful_code)
                final_answer = f"{final_result}\n\nHere is the code that was used to reach this solution:\n\n```python\n\n{successful_code_string}\n```"
                final_answer += '\n\nAre you satisfied with this result?'
                
                messages.append(AIMessage(content=final_answer))
                
    # update state
    state['messages'] = messages
    state['previous_node'] = 'Execute'
    state['code'] = successful_code_string

    return state
