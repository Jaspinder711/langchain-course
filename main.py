from dotenv import load_dotenv

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

llm = ChatOllama(model="llama3.2:latest", temperature=0)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 recent job postings on Linkedin for AI engineer in Bangalore ?")]})
    print(result)

if __name__ == "__main__":
    main()