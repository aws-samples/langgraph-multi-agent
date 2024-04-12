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
    """Use this tool to describe the plan created to solve a user's task.  You must always use this tool to describe the plan."""
    plan: str = Field(description="The exact plan that was generated to solve the user's task.")
    libraries: str = Field(description=f"A comma-separated list of pybaseball libraries that are used in the plan.  Possible pybaseball libraries are {', '.join(library_dict.keys())}")


# define language models
#llm_haiku = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
#llm_sonnet = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
llm_opus = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)

llm_initial_plan = llm_opus.bind_tools([InitialPlan])

def collect_library_helpers(libraries):
    '''
    Collect pybaseball library documentation
    '''
    
    print(f'Collecting documentation for {libraries}')
    lib_list =[i.strip() for i in libraries.split(',')]
    
    helper_string = ''
    for lib in lib_list:
        lib_detail = library_dict[lib]
        docs = lib_detail['docs']
        #data_dictionary = lib_detail['data_dictionary']
        helper_string += f'Text between the <{lib}_documentation></{lib}_documentation> tags is documentation for the {lib} library.  Consult this section to confirm which attributes to pass into the {lib} library.\n<{lib}_documentation>\n{docs}\n</{lib}_documentation>\n'
        #helper_string += f'Text between the <{lib}_dictionary></{lib}_dictionary> tags is the data dictionary for the {lib} library.  Consult this section to confirm which columns are present in the response from the {lib} library.\n<{lib}_dictionary>\n{data_dictionary}\n</{lib}_dictionary>'

    return helper_string

#   formulate
INITIAL_PLAN_SYSTEM_PROMPT = '''
<instructions>You are a world-class Python programmer and an expert on baseball, with a specialization in data analysis using the pybaseball Python library. 
Your expertise is in formulating plans to complete tasks related to baseball data analysis.  You provide detailed steps that can be executed sequentially to solve the user's task.
You always use the InitialPlan tool to describe the plan.
 
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
1. 'mlbam' is the ID that should be used to link players across tables.  Use the playerid_reverse_lookup pybaseball library to convert an mlbam to a player name.
2. Every step that includes a pybaseball function call should include the specific inputs required for that function call.
3. The last step of the plan should always include a print() statement to describe the results.
4. You must use the InitialPlan tool to describe the plan.
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
        ("user", "{task}.  You must use the InitialPlan tool to describe the plan."), 
    ])

    initial_plan_chain = initial_prompt | llm_initial_plan 

    result = initial_plan_chain.invoke({'task':task, 'existing_plan':existing_plan, 'similar_task':similar_task, 'libraries_string': libraries_string}, config=langchain_config)

    # parse the tool response
    tool_calls = result.tool_calls
    
    initial_plan = [t['args']['plan'] for t in tool_calls if t['name'] == 'InitialPlan'][0]
    pybaseball_libraries = [t['args']['libraries'] for t in tool_calls if t['name'] == 'InitialPlan'][0]

    return initial_plan, pybaseball_libraries


# main function
def node(state):
    print(f'\n*** Entered Initialize Node ***\n')
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
    
    # collect documentation on functions
    helper_string = collect_library_helpers(pybaseball_libraries)
    
    # update state
    state['plan'] = initial_plan
    state['previous_node'] = 'Initialize'
    state['function_detail'] = helper_string
    
    return state
            