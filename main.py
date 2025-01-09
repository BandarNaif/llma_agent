# Import relevant functionality
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from read_pdf_file import qa_chain_tool
from SearchTool import search_tool
# Create the agent
memory = MemorySaver() # To Save conversation
model =ChatOllama(model="mistral") # To Load LLM model
tools = [qa_chain_tool,search_tool] # to store all tools
agent_executor = create_react_agent(model, tools, checkpointer=memory) # Create agent

# Use the agent




def handle_conversation():
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("You : ")
        if query.lower()=="exit":
            break
        input_messages = [HumanMessage(query)]
        output = agent_executor.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()


if __name__=="__main__":
    handle_conversation()
    