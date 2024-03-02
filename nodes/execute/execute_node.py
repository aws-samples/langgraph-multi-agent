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
    
    ## State
    code = state["code"]
    iter = state["iterations"]
    
    print(f"Executing: {code}")
    # increment iterations
    iter += 1 

    # Attempt to execute code block
    result = python_repl.invoke(code)
    if result != '':
        print(f'Result: {result}')

    return {"result": result,
            "iterations": iter,
            "code": code}