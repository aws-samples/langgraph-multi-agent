# core libraries
import re

# langchain libraries
from langchain_core.messages import ToolMessage
from langchain_experimental.tools import PythonREPLTool

# define python repl
python_repl = PythonREPLTool()

# initiate python_repl to ignore warnings
python_repl.invoke('import warnings\nwarnings.simplefilter("ignore")')

def node(state):
    """
    Execute line of code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    # State
    session_id = state['session_id']
    messages = state['messages']
    successful_code = state['successful_code']
    
    # collect tool call metadata   
    code = [c['input']['code'] for c in messages[-1].content if c['type'] == 'tool_use'][0]
    tool_call_id = [c['id'] for c in messages[-1].content if c['type'] == 'tool_use'][0]
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    print(f"Executing: {code}")

    # Attempt to execute code block
    result = python_repl.invoke(code, config=langchain_config)
    
    if result != '':
        print(f'Result: {result}')
    
    '''
    if 'error' in result.lower():
        messages.append(ToolMessage(content=f'The previous step reached an error with the following code:\n\n```python\n{code}\n```\n\nHere was the error: {result}', tool_call_id=tool_call_id))
    else:
        messages.append(ToolMessage(content=f'The previous step completed successfully with the following code:\n\n```python\n{code}\n```\n\nHere was the result: {result}', tool_call_id=tool_call_id))
        successful_code.append(code)
    '''
    if 'error' in result.lower():
        messages.append(ToolMessage(content=f'The previous code reached an error.  Here was the error: {result}', tool_call_id=tool_call_id))
    else:
        successful_code.append(code)
        if result != '':
            messages.append(ToolMessage(content=f'The previous step completed successfully.  Here was the result: {result}', tool_call_id=tool_call_id))
        else:
            messages.append(ToolMessage(content=f'The previous step completed successfully.', tool_call_id=tool_call_id))
        
        
    return {'messages': messages, 'successful_code':successful_code}
