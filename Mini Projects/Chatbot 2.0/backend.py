from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, List

import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode

import sqlite3
import operator

import os
os.environ["LANGCHAIN_PROJECT"] = "Chatbot 2.0"

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    max_tokens=512,
    temperature=0,
    reasoning_effort="low"
)

# Tools
search_tool = DuckDuckGoSearchResults()

@tool
def calculator(first_number: float, second_number: float, operator: str) -> dict:
    """
    Performs a basic mathematical operation on 2 numbers based on the operator selected
    """
    match operator:
        case "+":
            result = first_number + second_number
        case "-":
            result = first_number - second_number
        case "*":
            result = first_number * second_number
        case "/":
            if second_number != 0:
                result = first_number + second_number
            else:
                return {"error": "Division by 0 not possible"}
        
    return {"first_number": first_number, "second_number": second_number, "operator": operator, "result": result}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetches latest stock price for a particular symbol (eg: 'AAPL', 'TSLA') by API calling
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=7TMC9BD1R1R6EL0L'
    r = requests.get(url)
    return r.json()

tool_list = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tool_list)

# State
class ChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    

def chatbot_qa(state: ChatbotState) -> dict:

    messages = state["messages"]

    prompt = f"""
    Answer the last human question asked in the messages: {messages}. Ensure well structured and sufficiently detailed answer. 
    """
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

tool_node = ToolNode(tools=tool_list)

graph = StateGraph(ChatbotState)
graph.add_node("chatbot_qa", chatbot_qa)

graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot_qa")
graph.add_conditional_edges("chatbot_qa", tools_condition)
graph.add_edge("tools", "chatbot_qa")


conn = sqlite3.connect(
    database="Mini Projects/Chatbot 2.0/chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    thread_set = set()
    for checkpoint in checkpointer.list(None):
        thread_set.add(checkpoint.config["configurable"]["thread_id"])
    return list(thread_set)


# CONFIG = {
#     "configurable": {
#         "thread_id": "thread_1"
#     }
# }

# input_state = {
#     "messages": [
#         HumanMessage("Explain XGBoost algorithm in Machine Learning.")
#     ]
# }

# output_state = chatbot.invoke(input_state, config=CONFIG)
# # print(output_state)

# for key, val in output_state.items():
#     print("")
#     print(key)
#     print(val)