from graphs import execute_graph
from langchain_core.messages import AIMessage


def node(state):
    # collect metadata from state
    plan = state['plan']
    task = state['task']
    function_detail = state['function_detail']
    session_id = state['session_id']

    inputs = {"plan": plan, "task": task, "function_detail":function_detail, "session_id": session_id, 'messages':[], 'successful_code': []}

    for s in execute_graph.graph.stream(inputs, {"recursion_limit": 100}):
        if "__end__"  in s:
            generation_content = s['__end__']['generation_result'].content
            if isinstance(generation_content, list):
                successful_code = s['__end__']['successful_code']
                final_result = generation_content[1]['input']['answer']
                plan = generation_content[1]['input']['plan']
        
                successful_code_string = '\n'.join(successful_code)
                final_answer = f"{final_result}\nHere is the code that was used to reach this solution:\n\n```python\n\n{successful_code_string}\n```"
                final_answer += '\n\nAre you satisfied with this result?'
            else:
                final_answer = generation_content
                successful_code_string = ''
                plan = ''

    return {"messages": [AIMessage(content=final_answer)], 
            "previous_node": "Execute", 
            "code": successful_code_string,
            'plan': plan
           }
