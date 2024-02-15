import ast
from agents import execute_agent
from langchain_core.messages import HumanMessage, SystemMessage

EXECUTE_PROMPT = '''
You are a highly skilled Python programmer, and your task is to assist the User in executing a Python-based plan. Follow these instructions carefully:

Steps:
1. Use the Python_REPL tool to execute the plan, execuring each step one at a time. Executing the steps one at a time ensures that if an error occurs, previously successful steps are not affected. Functions must be called exactly as outlined in the execution plan.
2. Continue with step 1 until the entire plan is executed. Address any errors by adjusting the code as needed.

Rules:
1. Import all necessary libraries at the start of your code.
2. Always use the 'key_mlbam' value as the player id
2. Follow the execution plan as described, without suggesting improvements until all steps have been attempted.
3. Always assign the result of a pybaseball function call to a variable.
4. Execute the plan one step at a time with individual calls to the Python_REPL tool.
4. Use print() when you want to display the final result to the User.
'''

def node(state):
    # add system message to input
    system = SystemMessage(content=EXECUTE_PROMPT)
    messages = [system] + state['messages']
    
    inputs = {"messages": messages}

    for s in execute_agent.graph.stream(inputs, {"recursion_limit": 100}):
        if "__end__" not in s:
            print(s)
            print("----")
        '''
        # this portion is not necessary - just useful for cleaning up the streaming of the executor node output
        if "__end__" not in s:
            if 'agent' in s:
                key = 'agent'
            else:
                key = 'action'
                
            message = s[key]['messages'][-1]
            content = message.content
            if content != '':
                print(content)
            else:
                try:
                    args = ast.literal_eval(message.additional_kwargs['function_call']['arguments'])['__arg1']
                    print(f'\nInvoking PythonREPLTool with: {args}\n')
                except:
                    pass
        '''
    
    # format response 
    response = ast.literal_eval(s['__end__']['messages'][-1].additional_kwargs['function_call']['arguments'])

    content =  response['answer'] + '\n\n' 
    if response['modifications'] != 'none':
        content += response['modifications'] + '\n\n' 
    content += 'Here is the Python code that was executed to solve this task:\n\n```\n' 
    content += response['python_code'] 
    content += '\n```\n\nAre you satisfied with this result?'
    del response['answer']
    del response['modifications']
    del response['python_code']
    
    return {"messages": [HumanMessage(content=content, additional_kwargs=response, name='Executor')]}