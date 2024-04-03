from dotenv import load_dotenv, find_dotenv
import re

from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

# read local .env file
_ = load_dotenv(find_dotenv())

# define language model
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
#model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
llm = BedrockChat(model_id=model_id, model_kwargs={'temperature': 0})

def extract_text_between_markers(text):
    '''Helper function to extract code'''
    start_marker='```python'
    end_marker='```'

    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text.content)
    return matches[0]


# Prompt
GENERATE_SYSTEM_PROMPT = """
You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your goal is to update a code block in order to resolve an error.

Text between the <task></task> tags is ultimate goal of the Python program.
<task>
{task} 
</task>

Text between the <successful_code></successful_code> tags is the code that has been executed successfully so far.
<successful_code>
{successful_code}
</successful_code>

Text between the <errored_code></errored_code> tags is the Python code that reached an error.
<errored_code>
```python
{code} 
```
</errored_code>

Review the error message and rewrite the code block that threw an error to resolve the issue. 

Return all python code between three tick marks like this:

```python
python code goes here
```
"""


def node(state):
    """
    Generate a code solution based on LCEL docs and the input question 
    with optional feedback from code execution tests 

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    # State
    code = state["code"]
    task = state["task"]
    error = state['result']
    successful_code = state['successful_code']
    session_id = state['session_id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    generate_prompt_template = ChatPromptTemplate.from_messages([
        ("system", GENERATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ]).partial(code=code, task=task, successful_code=successful_code)

    # Chain
    generate_chain = generate_prompt_template | llm | RunnableLambda(extract_text_between_markers)

    code_solution = generate_chain.invoke({"messages": [HumanMessage(content=f'Here is the error message:\n\n<error>\n{error}\n</error>')]}, config=langchain_config)

    return {"code": code_solution}