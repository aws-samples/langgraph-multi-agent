from graphs import execute_graph
from langchain_core.messages import AIMessage


def node(state):
    # collect metadata from state
    code = state['code']
    plan = state['plan']
    task = state['task']
    function_detail = state['function_detail']
    # split by double newlines
    code_lines = code.split('\n\n')

    # initiate empty list to collect successfully executed code
    successful_code = []

    for line in code_lines:
        successful_code_str = '```python\n' + '\n'.join(successful_code) + '\n```'
        inputs = {"code": line, "plan": plan, "task": task, "function_detail":function_detail, "iterations": 0, "successful_code": successful_code_str}

        for s in execute_graph.graph.stream(inputs, {"recursion_limit": 100}):
            if "__end__" not in s:
                if 'execute' in s:
                    if 'error' not in s['execute']['result'].lower():
                        successful_code.append(s['execute']['code'])

    # exctact final answer
    final_answer = s['__end__']['result']

    final_answer += f"\nHere is the code that was used to reach this solution:\n{successful_code_str}"
    final_answer += '\n\nAre you satisfied with this result?'

    return {"messages": [AIMessage(content=final_answer)], 
            "previous_node": "Execute", 
            "code": '\n'.join(successful_code)
           }
