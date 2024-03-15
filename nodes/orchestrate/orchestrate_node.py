from dotenv import load_dotenv, find_dotenv
import os

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import BedrockChat

# read local .env file
_ = load_dotenv(find_dotenv()) 

# define language model
#model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
llm = BedrockChat(model_id=model_id, model_kwargs={'temperature': 0})

POS_FEEDBACK_SYSTEM_PROMPT = '''
A user has been provided with a plan and asked if they approve.  Classify the user's response in one of two ways:

1 - The user approves (Y)
2 - The user has requested some modification (N)

Respond ONLY with either Y or N.
'''

pos_feedback_prompt_template = ChatPromptTemplate.from_messages([
    ("system", POS_FEEDBACK_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"), 
])

pos_feedback_chain = pos_feedback_prompt_template | llm

def get_pos_feedback_indicator(state):
    last_message = [state['messages'][-1]]
    response = pos_feedback_chain.invoke({'messages':last_message})
    response = response.content.lower().strip()

    if response in ['y', 'n']:
        return response
    else:
        raise ValueError


def node(state):
    '''Used to revise the propsed plan based on User feedback'''

    previous_node = state['previous_node']

    if previous_node == None:
        return {'next': 'Plan'}
    elif previous_node in ['Plan', 'Revise']:
        pos_feedback_indicator = get_pos_feedback_indicator(state)

        if pos_feedback_indicator == 'n':
            return {'next': 'Revise'}
        else:
            return {'next': 'Convert'}
    elif previous_node == 'Execute':
        pos_feedback_indicator = get_pos_feedback_indicator(state)

        if pos_feedback_indicator == 'n':
            return {'next': 'Revise'}
        else:
            return {'next': 'Memorize'}
    elif previous_node == 'Memorize':
        return {"messages": [AIMessage(content="Please initialize a new session for a new task")], 
                "next": "END"
               }