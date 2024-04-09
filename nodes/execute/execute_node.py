# core libraries
import re

# langchain libraries
from langchain_core.messages import AIMessage
from langchain_experimental.tools import PythonREPLTool

# define python repl
python_repl = PythonREPLTool()

# initiate python_repl to ignore warnings
python_repl.invoke('import warnings\nwarnings.simplefilter("ignore")')

def extract_python_code(text):
    '''Helper function to extract code'''
    start_marker='```python'
    end_marker='```'

    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text)
    return matches[0]

def extract_final_result(text):
    '''Helper function to extract final result'''
    start_marker='<final_result>'
    end_marker='</final_result>'

    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text)
    return matches[0]


def node(state):
    """
    Execute line of code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    # State
    result = state["result"]
    #iterations = state["iterations"]
    session_id = state['session_id']
    messages = state['messages']
    successful_code = state['successful_code']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    if '<final_result>' in result:
        final_result = extract_final_result(result)
        
        return {'final_result': final_result}
    
    else:
        code = extract_python_code(result)
        print(f"Executing: {code}")
        # increment iterations
        #iterations += 1 

        # Attempt to execute code block
        result = python_repl.invoke(code, config=langchain_config)
        if result != '':
            print(f'Result: {result}')
            
            
        if 'error' in result.lower():
            messages.append(AIMessage(content=f'The previous step reached an error with the following code:\n\n```python\n{code}\n```\n\nHere was the error: {result}'))
        else:
            messages.append(AIMessage(content=f'The previous step completed successfully with the following code:\n\n```python\n{code}\n```\n\nHere was the result: {result}'))
            successful_code.append(code)
            
        return {"result": result, 'messages': messages, 'successful_code':successful_code}
