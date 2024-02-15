
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# define language model
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

system_prompt = '''
Read on the conversation below and revise the plan based on User's feedback.

Be sure the response ends with "Are you satisfied with this plan?"
'''
def node(state):
    '''Used to revise the propsed plan based on User feedback'''
    revise_prompt = ChatPromptTemplate.from_messages(
                                                [
                                                    ("system", system_prompt),
                                                    MessagesPlaceholder(variable_name="messages"),
                                                ]
                                                )

    revise_chain = revise_prompt | llm

    result = revise_chain.invoke(state)

    return {"messages": [HumanMessage(content=result.content, name='Revisor')]}