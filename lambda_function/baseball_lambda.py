# core libraries
import pickle

# langchain
from langchain_core.messages import HumanMessage

# custom local libraries
from agents import orchestrate_agent

# read function metadata from disk
state_dict_path = 'dynamodb/state.pkl'
    
def execute_workflow(task, session_id):
    # convert input to HumanMessage
    human_message = HumanMessage(content=task, name='User')
    
    # load previous state
    try:
        with open(state_dict_path, 'rb') as file:
            state_dict = pickle.load(file)
    except:
        state_dict = {}
    
    # check for previous state
    if session_id in state_dict:
        messages = state_dict[session_id] + [human_message]
    else:
        messages = [human_message]
        
    # uncomment to "hotwire" function with messages
    '''
    messages = [HumanMessage(content='How many games did the Rays win in 2022?', name='User'),
    HumanMessage(content='To determine how many games the Rays won in 2022, you can follow this plan:\n\n1. Import the `schedule_and_record` function from the `pybaseball` library.\n2. Call the `schedule_and_record` function with the following arguments:\n   - `season`: 2022\n   - `team`: "TBR" (abbreviation for the Tampa Bay Rays)\n3. Assign the returned dataframe to a variable, let\'s say `ray_games`.\n4. Filter `ray_games` to include only the rows where the value in the "W/L" column is "W" (indicating a win).\n5. Get the number of rows in the filtered dataframe to determine the number of games the Rays won in 2022.\n\nBy following this plan, you will be able to determine the number of games the Rays won in 2022 using the `pybaseball` library.\n\nAre you satisfied with this plan?', name='Plan'),
    HumanMessage(content='be sure to include both walk-off and regular wins', name='User'),
    HumanMessage(content='To determine the total number of games the Rays won in 2022, including both walk-off and regular wins, you can revise the plan as follows:\n\n1. Import the `schedule_and_record` function from the `pybaseball` library.\n2. Call the `schedule_and_record` function with the following arguments:\n   - `season`: 2022\n   - `team`: "TBR" (abbreviation for the Tampa Bay Rays)\n3. Assign the returned dataframe to a variable, let\'s say `ray_games`.\n4. Filter `ray_games` to include only the rows where the value in the "W/L" column is "W" (indicating a win).\n5. Get the number of rows in the filtered dataframe to determine the number of regular wins.\n6. Filter `ray_games` to include only the rows where the value in the "W/L" column is "W-wo" (indicating a walk-off win).\n7. Get the number of rows in the filtered dataframe to determine the number of walk-off wins.\n8. Add the number of regular wins and walk-off wins together to get the total number of games the Rays won in 2022.\n\nBy following this revised plan, you will be able to determine the total number of games the Rays won in 2022, including both walk-off and regular wins, using the `pybaseball` library.\n\nAre you satisfied with this plan?', name='Revisor'),
    HumanMessage(content='looks good', name='User'),
    HumanMessage(content="The Rays won a total of 86 games in 2022.\n\nNone\n\nHere is the Python code that was executed to solve this task:\n\n```from pybaseball import schedule_and_record\nray_games = schedule_and_record(2022, 'TBR')\nregular_wins = ray_games[ray_games['W/L'] == 'W']\nwalk_off_wins = ray_games[ray_games['W/L'] == 'W-wo']\ntotal_wins = len(regular_wins) + len(walk_off_wins)\nprint(total_wins)```\n\nAre you satisfied with this result?", additional_kwargs={'plan': '1. Import the `schedule_and_record` function from the `pybaseball` library.\n2. Call the `schedule_and_record` function with the following arguments:\n   - `season`: 2022\n   - `team`: "TBR" (abbreviation for the Tampa Bay Rays)\n3. Assign the returned dataframe to a variable, let\'s say `ray_games`.\n4. Filter `ray_games` to include only the rows where the value in the "W/L" column is "W" (indicating a win).\n5. Get the number of rows in the filtered dataframe to determine the number of regular wins.\n6. Filter `ray_games` to include only the rows where the value in the "W/L" column is "W-wo" (indicating a walk-off win).\n7. Get the number of rows in the filtered dataframe to determine the number of walk-off wins.\n8. Add the number of regular wins and walk-off wins together to get the total number of games the Rays won in 2022.'}, name='Executor'),
    HumanMessage(content='yes, thanks', name='User')]
    '''
        
    # execute
    for s in orchestrate_agent.graph.stream(
        {
            "messages": messages
        }
    ):
        if "__end__" not in s:
            print(s)
            print("----")
            
    # update state
    state_dict[session_id] = s['__end__']['messages']
    
    # write to disk
    with open(state_dict_path, 'wb') as file:
        pickle.dump(state_dict, file)
    
    # collect response
    last_message = s['__end__']['messages'][-1].content
    
    return last_message