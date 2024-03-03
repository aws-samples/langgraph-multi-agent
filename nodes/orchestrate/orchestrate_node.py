import boto3
from dotenv import load_dotenv, find_dotenv
import os

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock

# read local .env file
_ = load_dotenv(find_dotenv()) 

# define env vars
access_key_id = os.environ['ACCESS_KEY_ID']
secret_access_key = os.environ['SECRET_ACCESS_KEY']

# define boto clients
bedrock_runtime_client = boto3.client('bedrock-runtime', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

# define language model
model_kwargs = {'temperature': .001, 'max_tokens_to_sample': 10000}
model_id = 'anthropic.claude-instant-v1'
llm = Bedrock(model_id=model_id, client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url='prod.us-west-2.dataplane.bedrock.aws.dev')

pos_feedback_prompt = '''

Human: A user has been provided with a plan and asked if they approve.  Classify the user's response in one of two ways:

1 - The user approves (Y)
2 - The user has requested some modification (N)

Respond ONLY with either Y or N.

Here is the response from the user:

###
{user_input}
###

Assistant:
'''

pos_feedback_prompt_template = PromptTemplate.from_template(pos_feedback_prompt)
pos_feedback_chain = pos_feedback_prompt_template | llm


def get_pos_feedback_indicator(state):
    last_message = state['messages'][-1].content
    response = pos_feedback_chain.invoke({'user_input':last_message})
    response = response.lower().strip()

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