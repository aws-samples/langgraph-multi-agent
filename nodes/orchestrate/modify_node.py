# core libraries
from typing import List

# langchain libraries
from langchain_anthropic import ChatAnthropic
#from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
    

# Define data models
class ModifiedPlan(BaseModel):
    """Modify a plan based on feedback from the user"""
    plan: str = Field(description="The modified plan after making changes requested by the user.")


# define language models
#llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
#llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_modify = llm_sonnet.bind_tools([ModifiedPlan])


MODIFY_SYSTEM_PROMPT = '''
<instructions>Review the original plan and update it based on the new request while maintaining the format.
If the original plan already aligns with the new request, return it without any modifications.
Before updating the plan, do some analysis within <thinking></thinking> tags. 
</instructions>

Text bewteen the <original_task></original_task> tags is the goal of the original plan.
<original_task>
{original_task}
</original_task>

Text bewteen the <original_plan></original_plan> tags is the original plan to be modified.
<original_plan>
{existing_plan}
</original_plan>
'''

def modify_existing_plan(task, nearest_task, nearest_plan, langchain_config): 
    '''Used to revise the propsed plan based on User feedback'''
    modify_prompt = ChatPromptTemplate.from_messages([
        ("system", MODIFY_SYSTEM_PROMPT),
        ("user", "{updates}.  Use ModifiedPlan to describe the plan.") 
    ])

    modify_chain = modify_prompt | llm_modify 

    result = modify_chain.invoke({'existing_plan':nearest_plan, 'original_task': nearest_task, 'updates':task}, config=langchain_config)

    # parse tool response
    modified_plan = [c['input']['plan'] for c in result.content if c['type'] == 'tool_use'][0]
    
    return modified_plan


# main function
def node(state):
    # collect the User's task from the state
    task = state['task']
    session_id = state['session_id']
    messages = state['messages']
    nearest_task = state['nearest_task']
    nearest_plan = state['nearest_plan']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    print('Modifying nearest plan with User input')
    new_plan = modify_existing_plan(task, nearest_task, nearest_plan, langchain_config)
    
    messages.append(AIMessage(content=new_plan))

    return {"messages": messages, 
            "task": task,
            "plan": new_plan,
            "previous_node": "Modify"
        }

            