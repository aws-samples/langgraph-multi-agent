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
    code = state["code"]
    iterations = state["iterations"]
    session_id = state['session_id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    print(f"Executing: {code}")
    # increment iterations
    iterations += 1 

    # Attempt to execute code block
    result = python_repl.invoke(code, config=langchain_config)
    if result != '':
        print(f'Result: {result}')

    return {"result": result,
            "iterations": iterations,
            "code": code}