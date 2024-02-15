import chromadb
import pandas as pd

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# define language model
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

def commit_to_memory(task: str, updated_plan: str) -> dict:
    "Use this tool when you need to commit a task to memory"
    
    execution_plan_path = 'vectordb/execution_plan.csv'
    try:
        steps_df = pd.read_csv(execution_plan_path)
        # collect distance
        chroma_client = chromadb.Client()
        plan_collection = chroma_client.get_collection(name='execution_plan')
        result = plan_collection.query(query_texts=task, n_results=1)
        distance = result['distances'][0][0]

        if distance > .01: # add new task
            steps_df.loc[len(steps_df.index)] = [task, updated_plan]
            response = 'Task commited to memory'
        else: # update existing task
            same_task = result['documents'][0][0]
            same_task_index = steps_df.index[steps_df['task'] == same_task].tolist()[0]
            steps_df.loc[same_task_index] = [task, updated_plan]
            response = 'Task updated in memory'

        steps_df.to_csv(execution_plan_path, index=False)
    except Exception as e:
        print(e)
        response = e
    return response

system_prompt = '''
Review the messages provided and extract the user's request related to baseball. Format your response as a JSON object with the key 'request', containing the exact request made by the user, including any necessary modifications for clarity or specificity. Use the following template for your response:

```json
{{"request": "<user_request>"}}
```

Ensure that the request is concise and directly related to baseball, and that it captures the essence of the user's inquiry without additional unrelated information. 

Example:
If the user's message is "I'm wondering how many home runs Babe Ruth hit," your response should be:

```json
{{"request": "How many home runs did Babe Ruth hit?"}}
```

Remember to:
- Include only the relevant details related to baseball in the user's request.
- Use the JSON object format with the key 'request' for your response.                                                  
'''

def node(state):
    '''Used to submit a successful execution to long term memory'''

    memorize_prompt = ChatPromptTemplate.from_messages(
                                                [
                                                    ("system", system_prompt),
                                                    MessagesPlaceholder(variable_name="messages"),
                                                ]
                                                )

    # collect the task from User
    memorize_chain = memorize_prompt | llm | JsonOutputParser()
    inputs = {'messages':state['messages']} 

    result = memorize_chain.invoke(inputs)

    task = result['request']

    # collect the final plan executed by the Executor
    last_execute_message = [m for m in state['messages'] if m.name == 'Executor'][-1]
    updated_plan = last_execute_message.additional_kwargs['plan']
    
    # commit
    commit_to_memory(task=task, updated_plan=updated_plan)
        
    return {"messages": [HumanMessage(content="Thank you, task has been written to memory", name='Memorizer')]}