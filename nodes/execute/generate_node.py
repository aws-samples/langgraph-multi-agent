#from langchain_community.chat_models import BedrockChat
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field


opus_model_id = 'claude-3-opus-20240229'
sonnet_model_id = 'claude-3-sonnet-20240229'
haiku_model_id = 'claude-3-haiku-20240307'

llm = ChatAnthropic(model=opus_model_id, temperature=0)

# define tools
class PythonREPL(BaseModel):
    """A Python REPL that can be used to executed Python code"""
    code: str = Field(description="Code block to be executed in a Python REPL")
    
class FinalAnswer(BaseModel):
    """Collects the final answer once the entire execution plan has been executed.  Always use this tool after the entire plan has been executed successfully."""
    answer: str = Field(description="The answer to the user's task.")
    plan: str = Field(description="An updated plan after any changes that were necessary during execution.  This plan should be identical to the original plan except updated with any changes that were necessary during execution.")

# bind tools to model
llm_with_tools = llm.bind_tools([PythonREPL, FinalAnswer])

# Prompt
GENERATE_SYSTEM_PROMPT = '''<instructions>You are a highly skilled Python programmer.  Your goal is to help a user execute a plan by writing code for the PythonREPL tool.</instructions>

As a reference, text between the <nearest_code></nearest_code> tags is the code that was used to solve a similar plan.
<nearest_code>
{nearest_code}
</nearest_code>

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
3. When executing the last step of the plan, use a print() statement to describe the results.
4. Comment your code liberally to be clear about what is happening and why.
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
    #nearest_plan = state['nearest_plan']
    nearest_code = state['nearest_code']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    # determine the next set of messages
    convert_message = 'Convert the next step of the plan into code that can be executed in a Python REPL.  Use the PythonREPL tool in your response.'
    troubleshoot_message = 'What information would be useful in order to troubleshoot this error?  Write Python code that can be executed in a python repl to confirm this information.  Use the PythonREPL tool in your response.'

    if len(messages) > 0 and messages[-1].content[:34] == 'The previous step reached an error':
        messages.append(HumanMessage(content=troubleshoot_message))
    else:
        messages.append(HumanMessage(content=convert_message))
    
    # define prompt template
    generate_prompt_template = ChatPromptTemplate.from_messages([
        ("system", GENERATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"), 
    ]).partial(function_detail=function_detail, task=task, plan=plan, nearest_code=nearest_code) #nearest_plan=nearest_plan,

    # define chain
    generate_chain = generate_prompt_template | llm_with_tools
    
    result = generate_chain.invoke({"messages": messages}, config=langchain_config)
    
    messages.append(result) # AIMessage type

    return {"generation_result": result, 'messages': messages}