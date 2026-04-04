import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # go up to parent

from utils.ssl_fix import apply_ssl_fix, get_http_client   
apply_ssl_fix()                                                

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv(find_dotenv())

@tool
def search(query: str) -> str:
    """Search for given query on internet."""
    print(f"searching for .. {query}")
    tavily = TavilyClient()
    return tavily.search(query=query)

def main():
    with get_http_client() as http_client:            
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            http_client=http_client                   
        )
        tools = [TavilySearch()] #[search,DuckDuckGoSearchRun()]
        agent = create_agent(model=llm,tools=tools)
        result = agent.invoke({"messages":HumanMessage(content="who is the best crickter")})
        #"search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
        print(result["messages"][-1].content)
    
if __name__ == "__main__":
    main()
