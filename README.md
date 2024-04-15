# Multi-Agent Data Analysis Assistant with LangGraph

## Overview

The purpose of this repository is to demonstrate how LangGraph can be used to build a stateless multi-agent workflow to serve as an assistant for data analysis.  This workflow leverages the [pybaseball](https://github.com/jldbc/pybaseball) Python library to extract data which is then used for analysis based on the user's request.

The workflow consists of two agent systems: one for general workflow and planning and another for code generation and execution.  An Agent is fundamentally a language model on a loop until some stopping condition is met, and a Graph is what we use to define the loop. For this reason, "Agent" and "Graph" are used interchangeably in this documentation.

![workflow](images/workflow.drawio.png  "Workflow")

The graphs are organized in a modular manner so that each node serves a specific purpose.

![nodes](images/nodes.drawio.png  "Nodes")

## Sample Usage

The sample notebooks demonstrate three simple use cases as examples of how this system can be used.  

```
1.  How many home runs did Derek Jeter hit in 2010?
```

```
2.  Plot the cumulative sum of strikeouts thrown by Danny Duffy in the 2018 season.
```

```
3.  Consider the first week of August 2020 - find 3 pitchers who's curveballs were most similar to Max Scherzer's.
```

## Getting Started
1. Clone repository and navigate to the `langgraph-multi-agent` folder 
2. Update local `env` file with the required environment variables and rename to `.env`

## Running the Jupyter Notebooks
1. Create and activate a virtual environment
```
python3 -m venv venv
```
```
source venv/bin/activate
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Open and execute the sample Jupyter Notebooks in order.  You must ensure that the Jupyter notebook is running a python kernel tied to the virtual environment you have created so that the required libraries will be available.  

The intent of the sample notebooks is to first demonstrate a use case that is unknown to the agent system, and then to demonstrate a slight variation of the same use case after the system has learned the pattern.

## Folder Structure

```
langgraph-multi-agent
│   README.md
│   .gitignore    
│   requirements.txt   
│   env  :  Template for creating local .env file  
│   *_sample_*.ipynb  :  Demonstration of a use case
│
└───images
│   │   workflow.drawio.png  
│   │   nodes.drawio.png 
│
└───graphs
│   │   __init__.py
│   │   execute_graph.py : Resposible for generating and executing Python code to execute the plan
│   │   orchestrate_graph.py : Resposible for general orchestration and plan creation
│
└───state
│   │   __init__.py
│   │   create_functions_statsapi.ipynb : Helper notebook to persist pybaseball function metadata
│   │   data_dictionary.py : String representations of the data dictionary for pybaseball functions
│   │   functions.json : Output from create_functions_statsapi.ipynb that will be read by the agent system
│
└───function
│   │   __init__.py
│   │   baseball_lambda.py : Entrypoint for the Agent system
│
└───nodes
│   └───execute
│   │   │   __init__.py
│   │   │   execute_node.py : Responsible for creating the Execute Node
│   │   │   generate_node.py : Responsible for creating the Generate Node
│   │   │   summarize_node.py : Responsible for creating the Summarize Node
│   │
│   └───orchestrate
│   │   │   execute_graph_node.py : Triggers the Execute Graph
│   │   │   initialize_node.py : Responsible for creating the Initialize Node
│   │   │   memorize_node.py : Responsible for creating the Memorize Node
│   │   │   modify_node.py : Responsible for creating the Modify Node
│   │   │   orchestrate_node.py : Responsible for creating the Orchestrate Node
│   │   │   retrieve_node.py : Responsible for creating the Retrieve Node
│   │   │   revise_node.py : Responsible for creating the Revise Node
│   │   │   update_node.py : Responsible for creating the Update Node
│
└───vectordb
│   │   __init__.py
│   │   create_execution_plan_vectordb_entries.ipynb : Helper notebook to create or clear execution plans
│   │   execution_plan.csv : Execution plans are written to disk so that they can be read by vector database
│   │   vectordb.py : Helper function for creating and retrieving the vector database collection for execution plans

```