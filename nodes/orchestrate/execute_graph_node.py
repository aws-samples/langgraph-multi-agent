from graphs import execute_graph
from langchain_core.messages import AIMessage


def node(state):
    # collect metadata from state
    plan = state['plan']
    task = state['task']
    function_detail = state['function_detail']
    session_id = state['session_id']
    nearest_code = state['nearest_code']

    inputs = {"plan": plan, 
              "task": task, 
              "function_detail": function_detail, 
              "session_id": session_id, 
              'messages':[], 
              'successful_code': [], 
              'nearest_code': nearest_code}

    for s in execute_graph.graph.stream(inputs, {"recursion_limit": 100}):
        if "__end__"  in s:
            print('END')
            print(s)

            successful_code = s['__end__']['successful_code']
            final_result = s['__end__']['messages'][-1].content
    
            successful_code_string = '\n'.join(successful_code)
            final_answer = f"{final_result}\n\nHere is the code that was used to reach this solution:\n\n```python\n\n{successful_code_string}\n```"
            final_answer += '\n\nAre you satisfied with this result?'

    return {"messages": [AIMessage(content=final_answer)], 
            "previous_node": "Execute", 
            "code": successful_code_string,
            'plan': plan
           }
