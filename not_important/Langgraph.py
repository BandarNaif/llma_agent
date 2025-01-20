from IPython.display import Image, display
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from not_important.read_pdf_file import qa_chain_tool

# add memory to remeber the previos conversation
memory = MemorySaver()
# Configration for memory
config = {"configurable": {"thread_id": "1"}}
# Class to handle the updated messages without overwriting


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Create state graph and make the input is the state
graph_builder = StateGraph(State)
# Create tools
tool = TavilySearchResults(max_results=2)
tools = [tool, qa_chain_tool]
# Loading model
llm = ChatOllama(model="mistral",temperature=0)
# Make chain between llm and tools
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)
# function for chatbot


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


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


while True:

    # code below for adding messages and conversation with AI
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()
