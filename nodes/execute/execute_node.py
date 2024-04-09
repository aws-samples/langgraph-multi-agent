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
    result = state["result"]
    session_id = state['session_id']
    messages = state['messages']
    successful_code = state['successful_code']
    
    print(result)
    # collect tool call metadata   
    code = result.content[1]['input']['code']
    tool_call_id = result.content[1]['id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    print(f"Executing: {code}")

    # Attempt to execute code block
    result = python_repl.invoke(code, config=langchain_config)
    if result != '':
        print(f'Result: {result}')
        
    if 'error' in result.lower():
        messages.append(ToolMessage(content=f'The previous step reached an error with the following code:\n\n```python\n{code}\n```\n\nHere was the error: {result}', tool_call_id=tool_call_id))
    else:
        messages.append(ToolMessage(content=f'The previous step completed successfully with the following code:\n\n```python\n{code}\n```\n\nHere was the result: {result}', tool_call_id=tool_call_id))
        successful_code.append(code)
    
    return {'messages': messages, 'successful_code':successful_code}
