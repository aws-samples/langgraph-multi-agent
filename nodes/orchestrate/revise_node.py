
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# define language model
llm = ChatOpenAI(model="gpt-4-turbo-preview",temperature=0)

REVISE_PROMPT = '''
You are world class Data Analyst and an expert on baseball and analyzing data through the pybaseball Python library.  
Your goal is to help a User create a plan that can be used to complete a task.

Review the task, the original plan, and the details related to the pybaseball functions in the plan.  Then revise the plan based on feedback from a User.

Here is the original plan:

{plan}

Here are details about the pybaseball functions in use.  Do not attempt to use any feature that is not explicitly listed in the data dictionary for that function.

{function_detail}

Here is the feedback:
{revision}

Be sure the response ends with "Are you satisfied with this plan?
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

    return {"messages": [HumanMessage(content=revised.content)], 
            "plan": revised.content, 
            "previous_node": "Revise" 
           }