from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv(override=True)
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# tavily = TavilyClient()

# @tool
# def search(query : str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answers")

llm = ChatOllama(model="llama3.2:latest", temperature=0)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt=("""You are a research assistant. Use TavilySearch to find information.
        When you have enough information, you MUST respond by calling the 
        AgentResponse tool with your final answer and source URLs. 
        Do NOT reply with plain text as your final answer."""))

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 recent job postings on Linkedin for AI engineer in Bangalore ?")]})
    print(result)

if __name__ == "__main__":
    main()