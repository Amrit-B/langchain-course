import os
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Tool that searches the internet for a query."""
    print(f"Searching for '{query}'...")
    return tavily.search(query=query)


api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

class Source(BaseModel):
    """Scheme for a source used by the agent"""
    url:str = Field(description = "The Url of the source")

class AgentResponse(BaseModel):
    """Scheme for the response of the agent"""
    answer:str = Field(description = "The Agent's answer to the question")
    sources:List[Source] = Field(default_factory= list, description = "The list of sources used by the agent to generate answer")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
)

tools = [search]
agent = create_agent(model=llm, tools=tools, response_format = AgentResponse)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details, Make sure its no more than 2 days old")})
    print(result)


if __name__ == "__main__":
    main()
