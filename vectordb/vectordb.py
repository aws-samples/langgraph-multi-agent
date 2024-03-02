import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    

def get_execution_plan_collection():
    '''
    Returns a Chroma collection

    params:
        NA

    returns:
        collection (Chroma): Chroma collection
    '''

    collection_name = 'execution_plan'

    # initiate a new collection
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=sentence_transformer_ef)

    # collect the evaluation table as the question bank
    steps_table = pd.read_csv('vectordb/execution_plan.csv')
    # double-check to eliminate duplicate tasks
    steps_table = steps_table.drop_duplicates(subset=['task'])
    steps_table = steps_table.reset_index(drop=True)

    task_list = []
    uuid_list = []
    metadata_list = []

    # add questions to the collection
    for i in range(steps_table.shape[0]):
        task = steps_table['task'][i]
        uuid = str(hash(task))
        plan = steps_table['plan'][i]
        code = steps_table['code'][i]


        uuid_list.append(uuid)
        task_list.append(task)
        metadata_list.append({"plan": plan, "code": code})

    # add texts to collection
    collection.add(
        documents=task_list,
        metadatas=metadata_list,
        ids=uuid_list
    )

    return collection