from dotenv import load_dotenv, find_dotenv

from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# read local .env file
_ = load_dotenv(find_dotenv())

# define language model
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
#model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
llm = BedrockChat(model_id=model_id, model_kwargs={'temperature': 0})


# Prompt
CONVERT_SYSTEM_PROMPT = '''<instructions>You are a highly skilled Python programmer.  Your goal is to help a user execute a plan by writing code for a Python REPL.</instructions>

Text between the <function_detail></function_detail> tags is documentation on the functions in use.  Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.
<function_detail> 
{function_detail}
</function_detail>

Text between the <task></task> tags is the goal of the plan.
<task>
{task}
</task>

Text between the <plan></plan> tags is the entire plan that will be executed.
<plan>
{plan}
</plan>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. Import all necessary libraries at the start of your code.
2. Always assign the result of a pybaseball function call to a variable.
3. When writing code for the last step in the plan, always use print() to write a detailed summary of the results.
4. Never write functions
5. Return all python code between three tick marks like this:
```python
python code goes here
```
6. Comment your code liberally to be clear about what is happening and why.
7. If the entire plan has been executed, answer the request between the <task></task> tags between <final_result></final_result> tags.
</rules>
'''



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
    plan = state['plan']
    task = state['task']
    messages = state['messages']
    session_id = state['session_id']
    function_detail = state['function_detail']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    # determine the next set of messages
    convert_message = 'Convert the next step of the plan into code that can be executed in a Python REPL.'
    troubleshoot_message = 'What information would be useful in order to troubleshoot this error?  Write Python code that can be executed in a python repl to confirm this information.'
    
    if len(messages) > 0 and messages[-1].content[:34] == 'The previous step reached an error':
        messages.append(HumanMessage(content=troubleshoot_message))
    else:
        messages.append(HumanMessage(content=convert_message))
    
    # define prompt template
    generate_prompt_template = ChatPromptTemplate.from_messages([
        ("system", CONVERT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ]).partial(function_detail=function_detail, task=task, plan=plan)

    # define chain
    generate_chain = generate_prompt_template | llm | StrOutputParser()
    
    result = generate_chain.invoke({"messages": messages}, config=langchain_config)

    return {"result": result, 'messages': messages}