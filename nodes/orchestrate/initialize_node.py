# core libraries=
import json

# langchain libraries
from langchain_anthropic import ChatAnthropic
#from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

with open('state/functions.json', 'r') as file:
    library_dict = json.load(file)
    
libraries_string = ''
for key in library_dict:
    docs = library_dict[key]['docs']
    libraries_string += f'<{key}>\n{docs}\n</{key}>\n'
    
# Define data model
class InitialPlan(BaseModel):
    """Initial plan generated to solve the user's task"""
    plan: str = Field(description="The exact plan that was generated to solve the user's task.")
    libraries: str = Field(description=f"A comma-separated list of pybaseball libraries that are used in the plan.  Possible pybaseball libraries are {', '.join(library_dict.keys())}")


# define language models
#llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
#llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_initial_plan = llm_opus.bind_tools([InitialPlan])

#   formulate
INITIAL_PLAN_SYSTEM_PROMPT = '''
<instructions>You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your expertise is in formulating plans to complete tasks related to baseball data analysis.  You provide detailed steps that can be executed sequentially to solve the user's task.

Before creating the plan, do some analysis within <thinking></thinking> tags.
</instructions>

Text between the <libraries></libraries> tags is the list of pybaseball libraries you may use, along with their documentation.
<libraries>
{libraries_string}
</libraries>

Text between the <similar_task></similar_task> tags is an example of a similar task to what you are being asked to evaluate.
<similar_task>
{similar_task}
</similar_task>

Text between the <similar_plan></similar_plan> tags is the plan that was executed for the similar task.
<similar_plan>
{existing_plan}
</similar_plan>

Text between the <rules></rules> tags are rules that must be followed.
<rules> 
1. 'mlbam' is the ID that should be used to link players across tables.
2. Every step that includes a pybaseball function call should include the specific inputs required for that function call.
3. The last step of the plan should always include a print() statement to describe the results.
</rules>
'''

def formulate_initial_plan(task, existing_plan, similar_task, langchain_config):
    """
    Formulate an initial plan to solve the user's task. 

    Arguments:
        - task (str): task from the user to be solved
        - existing_plan (str): plan associated with the nearest task
        - similar_task (str): nearest plan from the semanitic search
        - langchain_config (dict): configuration for the language model

    Returns:
        - initial_plan (str): Plan generated to solve the task
        - pybaseball_libraries (list): List of pybaseball libraries used in the plan
    """
    
    initial_prompt = ChatPromptTemplate.from_messages([
        ("system", INITIAL_PLAN_SYSTEM_PROMPT),
        ("user", "{task}.  Use InitialPlan to describe the plan."), 
    ])

    initial_plan_chain = initial_prompt | llm_initial_plan 

    result = initial_plan_chain.invoke({'task':task, 'existing_plan':existing_plan, 'similar_task':similar_task, 'libraries_string': libraries_string}, config=langchain_config)
    print('INITIAL CHAIN RESULT')
    print(result)
    # parse the tool response
    initial_plan = [c['input']['plan'] for c in result.content if c['type'] == 'tool_use'][0]
    pybaseball_libraries = [c['input']['libraries'] for c in result.content if c['type'] == 'tool_use'][0]

    return initial_plan, pybaseball_libraries


# main function
def node(state):
    # collect the User's task from the state
    task = state['task']
    session_id = state['session_id']
    nearest_task = state['nearest_task']
    nearest_plan = state['nearest_plan']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}

    # no close plan - formulate a one
    print('Formulating a new plan to solve the task.')
    
    # forumalte an initial plan
    initial_plan, pybaseball_libraries = formulate_initial_plan(task, nearest_plan, nearest_task, langchain_config)
    
    return {"plan": initial_plan,
            "previous_node": "Initialize",
            'pybaseball_libraries': pybaseball_libraries
        }
            