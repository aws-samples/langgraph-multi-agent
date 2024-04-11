# core libraries
from typing import List

# langchain libraries
from langchain_anthropic import ChatAnthropic
#from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_core.pydantic_v1 import BaseModel, Field
    

# define language models
llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)

llm_summarize = llm_haiku


SUMMARIZE_SYSTEM_PROMPT = '''
Use the context provided by the user to respond to this request:

<request>
{task}
</request>

Do not mention the context in your response.  Provide the minimum response necessary in order to address the request.
'''

def summarize_results(task, result, langchain_config): 
    '''Used to to summarize the result after the plan has been executed successfully'''
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZE_SYSTEM_PROMPT),
        ("user", "{result}")
    ])

    summarize_chain = summarize_prompt | llm_summarize 

    result = summarize_chain.invoke({'task':task, 'result': result}, config=langchain_config)
    
    return result

# main function
def node(state):
    # collect the User's task from the state
    task = state['task']
    session_id = state['session_id']
    messages = state['messages']
    result = messages[-1].content
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    print('Summarizing execution result')
    summary = summarize_results(task, result,langchain_config)
    
    messages.append(summary)
    
    # update state
    state['messages'] = messages

    return state

            