from dotenv import load_dotenv, find_dotenv

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

def extract_plan(message):
    '''
    Helper function to extract the final plan
    '''
    text = message.content
    start_tag = '<plan>'
    end_tag = '</plan>'
    start_index = text.find(start_tag)
    end_index = text.find(end_tag, start_index + len(start_tag))
    if start_index != -1 and end_index != -1:
        return text[start_index + len(start_tag):end_index]
    else:
        return None  # Return None if tags are not found


REVISE_SYSTEM_PROMPT = '''
You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a User create a plan that can be used to complete a task.

Review the original plan and the details related to the pybaseball functions in the plan.  Then revise the plan based on feedback from a User.

Text between the <original_plan></original_plan> tags is the original plan to be revised.
<original_plan>
{plan}
</original_plan>

Text bewteen the <function_detail></function_detail> tags is information about the functions in use.  
<function_detail> 
{function_detail}
<function_detail>

Text between the <rules></rules> tags are rules that must be followed.
<rules>
1. Always return the plan between <plan></plan> tags 
2. Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.
3. Every step that includes a pybaseball function call should include the specific input required for that function call
</rules>
'''

revise_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REVISE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"), 
])

revision_chain = revise_prompt_template | llm  | RunnableLambda(extract_plan)


def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    # collect metadata from state
    plan = state['plan']
    function_detail = state['function_detail']
    messages = state['messages']
    session_id = state['session_id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    # invoke revise chain
    revised = revision_chain.invoke({'plan': plan, 'function_detail': function_detail, 'messages': [messages[-1]]}, config=langchain_config)
    revised += '\n\nAre you satisfied with this plan?'
    
    return {"messages": [HumanMessage(content=revised)], 
            "plan": revised, 
            "previous_node": "Revise" 
           }