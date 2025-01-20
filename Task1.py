# Import Laiberies 
from typing import Annotated
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import pandas as pd
import folium
from itertools import cycle
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.agents import create_csv_agent



# Lodaing Ollama model (Mistral)
llm = ChatOllama(model="mistral",temperature=0)

# add memory to remeber the previos conversation
memory = MemorySaver()

# Configration ID for memory
config = {"configurable": {"thread_id": "1"}}

# Class to handle the updated messages without overwriting
class State(TypedDict):
    messages: Annotated[list, add_messages]
    

# Tools 
#Tool 1 for getting columns for csv file
@tool
def get_columns(path: str):
    """
    -If you read get columns use this tool.
    Read csv files and get the names of the columns.
    
    Parameters:
        path: str - Path to the CSV file.
    
    Returns:
        list: List of the column names.
    """
    
    print(pd.read_csv(path).columns)
    return list(pd.read_csv(path).columns)

# Tool 2 is visualize and create file HTNL
@tool
def visualize_geo_points(path, category_column, lat_column='lat', lon_column='lon', output_file='map.html'):
    """
    -If you read Visualize use this tool.
    Visualize geospatial points on a map with category-based pin colors.
    
    Parameters:
        path (str): Path to CSV file.
        category_column (str): Column name for categories (e.g., 'type').
        lat_column (str): Column name for latitude values.
        lon_column (str): Column name for longitude values.
        output_file (str): Name of the output HTML file.
    
    Returns:
        str: Path to the generated map HTML file.
    """
    df = pd.read_csv(path)
    # Predefined color palette
    predefined_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue'
    ]

    # Assign colors dynamically to unique categories
    unique_categories = df[category_column].unique()
    color_cycle = cycle(predefined_colors)  # Cycle through colors if categories > colors
    color_map = {category: next(color_cycle) for category in unique_categories}

    # Create a Folium map centered on the mean location
    center_lat = df[lat_column].mean()
    center_lon = df[lon_column].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    # Add points to the map with dynamic colors
    for _, row in df.iterrows():
        # Generate a dynamic popup text
        popup_content = "<br>".join(
            [f"{col}: {row[col]}" for col in df.columns if col not in [lat_column, lon_column]]
        )

        folium.Marker(
            location=[row[lat_column], row[lon_column]],
            popup=popup_content,
            tooltip=row[category_column],
            icon=folium.Icon(color=color_map.get(row[category_column], 'gray'))  # Default to 'gray'
        ).add_to(m)

    # Save map to file
    m.save(output_file)
    return f"Map saved to {output_file}"
# Tool 3 is for reading from pdf file 
def BandarInfo(query:str):
    """
    -If you read Bandar use this tool.
    Read pdf file and answer any question related to bandar.
    
    Parameters:
    query (str): question 
    
    Returns:
        Answer for question from pdf document.
    """

    # File loader
    document = PyPDFLoader("./data/Bandar.pdf").load()
    # Embedding model
    embed_model = OllamaEmbeddings(model="llama3")
    # Vector store index
    vectorstore = FAISS.from_documents(document, embed_model)
    # build chain to be ready for agent (Prepere the tool)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever())
    return qa_chain.run(query)


# Read CSV file and make csv agent answer and do action as needed
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
    response =csv_agent.run(query)
    return response



# List and tool and make chain with model 
tools = [get_columns,visualize_geo_points,BandarInfo,Read_csv]
llm_with_tools = llm.bind_tools(tools=tools)
tool_node = ToolNode(tools=tools)


# Function for Create ChatBot

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Create state graph and make the input is the state
graph_builder = StateGraph(State)
# add chatbot to graph
graph_builder.add_node("chatbot", chatbot)
# add tool to graph
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)


# code below for adding messages and conversation with AI
user_input = "Get columns of /home/pc/Desktop/llma_agent/data/wildLifeInKsa.csv and visualize it"
events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
for event in events:
    event["messages"][-1].pretty_print()