from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# define language model
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

pos_feedback_prompt = '''
A user has been provided with a plan and asked if they approve.  Classify the user's response in one of two ways:

1 - The user approves (Y)
2 - The user has requested some modification (N)
  
Respond ONLY with either Y or N.

Here is the response from the user:

###
{user_input}
###
'''

pos_feedback_prompt_template = PromptTemplate.from_template(pos_feedback_prompt)
pos_feedback_chain = pos_feedback_prompt_template | llm

def get_pos_feedback_indicator(state):
    last_message = state['messages'][-1].content
    response = pos_feedback_chain.invoke({'user_input':last_message})
    response = response.content.lower() 
    
    if response in ['y','n']:
        return response
    else:
        raise ValueError

def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    

    previous_node = state['previous_node']
        
    if previous_node == None:
        return {'next':'Plan'}
    elif previous_node in ['Plan', 'Revise']:
        pos_feedback_indicator = get_pos_feedback_indicator(state)

        if pos_feedback_indicator == 'n':
            return {'next':'Revise'}
        else:
            return {'next':'Convert'}
    elif previous_node == 'Execute':
        pos_feedback_indicator = get_pos_feedback_indicator(state)

        if pos_feedback_indicator == 'n':
            return {'next':'Revise'}
        else:
            return {'next':'Memorize'}
    elif previous_node == 'Memorize':
        return {"messages": [AIMessage(content="Please initialize a new session for a new task")], 
                "next": "END"
               }
        
 