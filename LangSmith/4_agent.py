from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.prebuilt import create_react_agent

import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent"


# Search tool
search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city.
    """

    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={os.getenv('WEATHERSTACK_API_KEY')}"
        f"&query={city}"
    )

    response = requests.get(url)
    response.raise_for_status()

    return json.dumps(response.json())


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# Create ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data]
)


# Invoke
response = agent.invoke(
    {
        "messages": [
            (
                "user",
                "Recent news about Nvidia"
            )
        ]
    }
)


# Print final answer
print(response["messages"][-1].content)