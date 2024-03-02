import re

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# custom local libraries
from vectordb import vectordb

# set a distance threshold for when to create a new plan vs modify an existing plan
threshold = .5

def extract_text_between_markers(text):
    '''Helper function to extract code'''
    start_marker='```python'
    end_marker='```'
    
    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(text.content)
    return matches[0]


CONVERT_PROMPT = '''
You are a highly skilled Python programmer.  Convert the User's plan into code that can be executed in a Python REPL.  Comment your code liberally to be clear about what is happening and why.

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

Now, Here is the user's plan you must convert:
###
{plan}
###

Rules:
1. Import all necessary libraries at the start of your code.
2. Always assign the result of a pybaseball function call to a variable.
3. Use print() when you want to display the final result to the User.
4. Never write functions
'''

# define language model
#llm = ChatOpenAI(model="gpt-4-turbo-preview",temperature=0)
llm = ChatOpenAI(model="gpt-4",temperature=0)

convert_prompt_template = PromptTemplate.from_template(CONVERT_PROMPT)
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
    
    # invoke convert chain
    response = convert_chain.invoke({'plan':plan, 'closest_plan': closest_plan, 'closest_code': closest_code})

    return {"code": response} 