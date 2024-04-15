from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat


# define language model
#model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
llm = BedrockChat(model_id=model_id, model_kwargs={'temperature': 0})

POS_FEEDBACK_SYSTEM_PROMPT = '''
If the user's message can be understood as positive affirmation with no additional information or clarifications, respond Y.
If there is any additional information in the user's message beyond a single positive affirmation, respond N.

Examples of positive affirmation may include but are not limited to "yes", "yep", "looks good", "great", "perfect"

Respond ONLY with either Y or N
'''

pos_feedback_prompt_template = ChatPromptTemplate.from_messages([
    ("system", POS_FEEDBACK_SYSTEM_PROMPT),
    ("user", "{message}")
])

pos_feedback_chain = pos_feedback_prompt_template | llm

def get_pos_feedback_indicator(state):
    # parse state metadata
    last_message = [state['messages'][-1]]
    session_id = state['session_id']
    
    # create langchain config
    langchain_config = {"metadata": {"conversation_id": session_id}}
    
    response = pos_feedback_chain.invoke({'message':last_message}, config=langchain_config)
    response = response.content.lower().strip()

    if response in ['y', 'n']:
        return response
    else:
        raise ValueError


def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    print(f'\n*** Entered Orchestrate Node ***')
    
    previous_node = state['previous_node']

    if previous_node == None:
        return {'next': 'Retrieve'}
    elif previous_node in ['Update', 'Revise', 'Modify']:
        pos_feedback_indicator = get_pos_feedback_indicator(state)

        if pos_feedback_indicator == 'n':
            return {'next': 'Revise'}
        else:
            return {'next': 'Execute'}
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