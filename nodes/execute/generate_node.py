import boto3
from dotenv import load_dotenv, find_dotenv
import os
import re

from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# read local .env file
_ = load_dotenv(find_dotenv())

# define env vars
access_key_id = os.environ['ACCESS_KEY_ID']
secret_access_key = os.environ['SECRET_ACCESS_KEY']

# define boto clients
bedrock_runtime_client = boto3.client('bedrock-runtime', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

# define language model
model_kwargs = {'temperature': .001, 'max_tokens_to_sample': 10000}
endpoint_url = 'prod.us-west-2.dataplane.bedrock.aws.dev'

llm = Bedrock(model_id='anthropic.claude-instant-v1', client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url=endpoint_url)


def extract_text_between_markers(text):
    '''Helper function to extract code'''
    start_marker='```python'
    end_marker='```'

    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text)
    return matches[0]


# Prompt
GENERATE_PROMPT = """

Human: You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your goal is to update a code block in order to resolve the error.
This code block is one step toward accomplishing this task:
\n ------- \n
{task} 
\n ------- \n
Here is the plan that is being followed to accomplishing this task:
\n ------- \n
{plan} 
\n ------- \n
Here is the code that has been executed successfully so far:
{successful_code} 
Here is documentation on the pybaseball function in use: 
\n ------- \n
{function_detail} 
\n ------- \n
Here is the next code block you attempted to execute: 
```python
{code} 
```
Here is the error message that was generated: 
\n ------- \n
{error} 
\n ------- \n
Rewrite this code block in order to resolve the error.

Return all python code between three tick marks like this:

```python
python code goes here
```

Assistant:
"""

generate_prompt_template = PromptTemplate.from_template(GENERATE_PROMPT)

# Chain
generate_chain = generate_prompt_template | llm | RunnableLambda(extract_text_between_markers)


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
    plan = state["plan"]
    error = state['result']
    successful_code = state['successful_code']
    function_detail = state["function_detail"]

    code_solution = generate_chain.invoke({"code": code,
                                    "task": task,
                                    "plan": plan,
                                    "error": error,
                                    "successful_code": successful_code,
                                    "function_detail": function_detail
                                    })
    return {"code": code_solution}