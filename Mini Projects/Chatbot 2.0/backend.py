from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, List

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

class EvalStructuredOutput(BaseModel):
    score: float = Field(description="Score of the answer based on given parameter", ge=0, le=10)
    justification: str = Field(description="Justification behind the score")

eval_llm = llm.with_structured_output(EvalStructuredOutput, method="json_schema")

# State
class ChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    correctness: str
    language: str
    relevance: str
    scores: Annotated[List[float], operator.add]
    summary: str
    

def chatbot_qa(state: ChatbotState) -> dict:

    messages = state["messages"]

    prompt = f"""
    Answer the last human question asked in the messages: {messages}. Ensure well structured and sufficiently detailed answer. 
    """
    response = llm.invoke(prompt)
    return {"messages": [response]}

def judge_correctness(state: ChatbotState) -> dict:

    messages = state["messages"]
    prompt = f"""
    Provide a numerical score between 1-10 and a short justification for the following conversation based on correctness.
    Judge any concepts/facts/figures/statements 
    Ensure you only judge last AI and Human conversation:
    {messages}.
    Return the response in JSON format with exactly these fields: `score` and `justification`
    """

    response = eval_llm.invoke(prompt)
    return {"scores": [response.score], "correctness": response.justification}    

def judge_language(state: ChatbotState) -> dict:

    messages = state["messages"]
    prompt = f"""
    Provide a numerical score between 1-10 and a short justification for the following conversation based on language.
    Judge on the basis of language, grammer, etc.
    Ensure you only judge last AI and Human conversation:
    {messages}.
    Return the response in JSON format with exactly these fields: `score` and `justification`
    """

    response = eval_llm.invoke(prompt)
    return {"scores": [response.score], "language": response.justification}  

def judge_relevance(state: ChatbotState) -> dict:

    messages = state["messages"]
    prompt = f"""
    Provide a numerical score between 1-10 and a short justification for the following conversation based on relevance.
    Judge any whether answer is relevant to the question or not.
    Ensure you only judge last AI and Human conversation:
    {messages}.
    Return the response in JSON format with exactly these fields: `score` and `justification`
    """

    response = eval_llm.invoke(prompt)
    return {"scores": [response.score], "relevance": response.justification}  

def final_eval(state: ChatbotState) -> dict:

    messages = state["messages"]

    prompt = f"""
    Give a final summary of the evaluation based on scores and justifications. Ensure you only check last AI and Human messages

    Messages: 
    {messages}.

    Justifications:
    Correctness: {state['correctness']} 
    Language: {state['language']}
    Relevance: {state['relevance']}
    Scores: {state['scores']}
    """

    response = llm.invoke(prompt)
    return {"summary": response.content}



graph = StateGraph(ChatbotState)
graph.add_node("chatbot_qa", chatbot_qa)
graph.add_node("judge_correctness", judge_correctness)
graph.add_node("judge_language", judge_language)
graph.add_node("judge_relevance", judge_relevance)
graph.add_node("final_eval", final_eval)

graph.add_edge(START, "chatbot_qa")

graph.add_edge("chatbot_qa", "judge_correctness")
graph.add_edge("judge_correctness", "judge_language")
graph.add_edge("judge_language", "judge_relevance")
graph.add_edge("judge_relevance", "final_eval")

graph.add_edge("final_eval", END)

conn = sqlite3.connect(
    database="Mini Projects/Chatbot 2.0/chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {
    "configurable": {
        "thread_id": "thread_1"
    }
}

input_state = {
    "messages": [
        HumanMessage("Explain XGBoost algorithm in Machine Learning.")
    ]
}

output_state = chatbot.invoke(input_state, config=CONFIG)
# print(output_state)

for key, val in output_state.items():
    print("")
    print(key)
    print(val)