import boto3
from dotenv import load_dotenv, find_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
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
endpoint_url = 'prod.us-west-2.dataplane.bedrock.aws.dev'

llm = Bedrock(model_id='anthropic.claude-v2:1', client=bedrock_runtime_client, model_kwargs=model_kwargs, streaming=True, endpoint_url=endpoint_url)


REVISE_PROMPT = '''

Human: You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a User create a plan that can be used to complete a task.

Review the task, the original plan, and the details related to the pybaseball functions in the plan.  Then revise the plan based on feedback from a User.

Here is the original plan:

{plan}

Here are details about the pybaseball functions in use.  Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.

{function_detail}

Here is the feedback:
{revision}

Be sure the response ends with "Are you satisfied with this plan?"

Assistant:
'''


revise_prompt_template = PromptTemplate.from_template(REVISE_PROMPT)
revision_chain = revise_prompt_template | llm 


def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    # collect metadata from state
    plan = state['plan']
    function_detail = state['function_detail']
    messages = state['messages']

    # collect revision request from the messages
    revision = messages[-1].content

    # invoke revise chain
    revised = revision_chain.invoke({'plan': plan, 'function_detail': function_detail, 'revision': revision})

    return {"messages": [HumanMessage(content=revised)], 
            "plan": revised, 
            "previous_node": "Revise" 
           }