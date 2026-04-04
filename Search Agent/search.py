import os
import sys
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
from pydantic import BaseModel, Field
from typing import List

load_dotenv(find_dotenv())

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class ResponseType(BaseModel):
    """Schema for agent response with response and sources"""
    response: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

@tool
def search(query: str) -> str:
    """Search for given query on internet."""
    print(f"searching for .. {query}")
    tavily = TavilyClient()
    return tavily.search(query=query)

# @tool
# def format_response(response: str, sources: List[Source]) :
#     """Call this tool to provide the final answer to the user with sources."""
#     return {"response": response, "sources": sources}

def main():
    with get_http_client() as http_client:            
        llm = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)
        tools = [TavilySearch()] #[search,DuckDuckGoSearchRun()]
        agent = create_agent(model=llm,tools=tools,response_format=ResponseType)
        user_input = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?" #input("ask anything : ")
        result = agent.invoke({"messages":HumanMessage(content=user_input)})
        #"search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
        print(result["messages"][-1].content)
    
if __name__ == "__main__":
    main()
