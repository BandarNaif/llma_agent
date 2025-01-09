from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=2) # Building tool

search_tool = Tool(
    name="Search online",
    func=search.run,
    description="Use this tool to search from online and trusted sources, dont use this tool until you see word search"
)