# import pandas as pd
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.agents import Tool


# # Read file
# df = pd.read_csv("./data/DroneResult.csv")
# # Load Embed model
# embed = OllamaEmbeddings(model="llama3")
# # Make the pandas ready for langchain functions
# documents = []
# for _, row in df.iterrows():
#     document = " | ".join(f"{col}: {row[col]}" for col in df.columns)
#     documents.append(document)
# # Split the data to chunk
# # Step 3: Split the data into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500, chunk_overlap=50)
# docs = text_splitter.create_documents(documents)

# vector_store = FAISS.from_documents(docs, embed)


# # Step 5: Define the Tool
# def query_vector_store(query):
#     """
#     Query the vector store to retrieve similar chunks based on the input query.
#     """
#     return vector_store.similarity_search(query, k=23)


# # Step 5: Define the Tool
# drone_result_tool = Tool(
#     name="Drone Result Query Tool",
#     func=query_vector_store,
#     description=(
#         "This tool answers questions based on drone result data from the dataset. "
#         "The dataset includes columns such as 'image_name', 'db_name', 'length', 'score', 'distance', and 'time'. "
#         "based on the query start search in datset and give the final result"
#     )
# )


import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_csv_agent
from langchain.agents import Tool


model = OllamaLLM(model="llama3")
csv_agent = create_csv_agent(
    model,
    './data/DroneResult.csv',
    verbose=True,
    allow_dangerous_code=True
)
csv_tool = Tool(
    name="Drone Result",
    func=csv_agent.run,
    description="This tool answers questions based on drone result data from the dataset. "
    "The dataset includes columns such as 'image_name', 'db_name', 'length', 'score', 'distance', and 'time'. "
    "based on the query start search in datset and give the final result")


def Read_csv(query:str,path:str):
    """
    -If you read read use this tool.
    Read csv file and answer any question related to csv file.
    
    Parameters:
    path (str) : Path of csv folder.
    query (str): question 
    
    Returns:
        Answer for question from csv document.
    """
    model = OllamaLLM(model="llama3")

    csv_agent = create_csv_agent(
                                model,
                                path,
                                verbose=True,
                                allow_dangerous_code=True)
    return csv_agent.run(query)