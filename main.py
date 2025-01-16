from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PDF_Tool import pdf_tool
from CSV_Tool import csv_tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Define the PromptTemplate
prompt = PromptTemplate(
    template="""
    You are an AI agent responsible for choosing the right tool to answer the user's query and dont use interent or online search. Below is the conversation so far:

    {messages}

    Based on the user's latest query, decide which tool to use:
    
    - If the query is related to Bandar, use the `pdf_tool` to extract relevant information from a PDF document.
    - If the query is related to Drones, use the `csv_tool` to extract relevant information from a CSV file.
    
    after using the right tool answer based on information that you got from the tool.
    """,
    input_variables=["messages"]
)

# Load the LLM model
llm = ChatOllama(model='mistral')

# Define tools
tools = [pdf_tool, csv_tool]

# Connect the LLM with the prompt
llm_prompt = LLMChain(llm=llm, prompt=prompt)

# Define memory to remember the chat history
memory = MemorySaver()
# Create the react agent
agent = create_react_agent(
    tools=tools,
    model=llm,
    state_modifier=prompt,
    checkpointer=memory
)

# Use the agent


def handle_conversation():
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("You : ")
        if query.lower() == "exit":
            break
        input_messages = [HumanMessage(query)]
        output = agent.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    handle_conversation()
