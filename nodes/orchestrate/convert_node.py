from dotenv import load_dotenv, find_dotenv
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import BedrockChat

# custom local libraries
from vectordb import vectordb

# read local .env file
_ = load_dotenv(find_dotenv()) 

# define language model
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
#model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
llm = BedrockChat(model_id=model_id, model_kwargs={'temperature': 0})

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5


def extract_text_between_markers(text):
    '''Helper function to extract code'''
    start_marker = '```python'
    end_marker = '```'

    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text.content)
    return matches[0]


CONVERT_SYSTEM_PROMPT = '''You are a highly skilled Python programmer.  Convert the User's plan into code that can be executed in a Python REPL.  Comment your code liberally to be clear about what is happening and why.

Return all python code between three tick marks like this:

```python
python code goes here
```

As a reference, here is a similar plan to what you will be asked to convert:
###
{closest_plan}
###

Here is the python code that was generated for this plan:
```python
{closest_code}
```

Rules:
1. Import all necessary libraries at the start of your code.
2. Always assign the result of a pybaseball function call to a variable.
3. Use print() when you want to display the final result to the User.
4. Never write functions
'''

convert_prompt_template = ChatPromptTemplate.from_messages([
    ("system", CONVERT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"), 
])

convert_chain = convert_prompt_template | llm | RunnableLambda(extract_text_between_markers)

def node(state):
    # parse state metadata
    task = state['task']
    plan = state['plan']

    # retrieve a collection on plans from vectordb
    plan_collection = vectordb.get_execution_plan_collection()

    # collect the closest plan for the task
    plan_result = plan_collection.query(query_texts=[task], n_results=1, include=['distances','metadatas','documents'])

    closest_plan = plan_result['metadatas'][0][0]['plan']
    closest_code = plan_result['metadatas'][0][0]['code']

    messages = [HumanMessage(content=plan)]

    # invoke convert chain
    response = convert_chain.invoke({'messages':messages, 'closest_plan': closest_plan, 'closest_code': closest_code})

    return {"code": response}