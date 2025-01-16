from langchain.prompts import PromptTemplate
# Define the PromptTemplate
prompt = PromptTemplate(
    template="""
    You are an AI agent responsible for choosing the right tool to answer the user's query and dont use interent or online search. Below is the conversation so far:

    {messages}

    Based on the user's latest query, decide which tool to use:
    
    - If the query is related to Bandar, use the `pdf_tool` to extract relevant information from a PDF document.
    - If the query is related to Drones, use the `csv_tool` to extract relevant information from a CSV file.
    
    Respond accordingly with the requested information.
    """,
    input_variables=["messages"]
)
