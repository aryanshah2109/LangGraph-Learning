import streamlit as st
from backend import chatbot, retrieve_all_threads
from uuid import uuid4
from langchain_core.messages import HumanMessage

## UTILITY FUNCTIONS
def generate_thread():
    return str(uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["all_threads"]:
        st.session_state["all_threads"].append(thread_id)

def reset_chat():
    new_thread_id = generate_thread()
    
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []
    add_thread(new_thread_id)

def get_messages(thread_id):
    state = chatbot.get_state(config = {
        "configurable": {
            "thread_id" : thread_id
        }
    })

    return state.values.get("messages", "")

def answer_stream():

    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "chatbot_qa":
            yield message_chunk.content

## SESSION MANAGEMENT
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread()

if "all_threads" not in st.session_state:
    st.session_state["all_threads"] = retrieve_all_threads()

# add first thread
add_thread(st.session_state["thread_id"])

## SIDEBAR
st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My chats")

for thread_id in st.session_state["all_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = get_messages(thread_id=thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            content = msg.content
            temp_messages.append({"role": role, "content": content})

        st.session_state["message_history"] = temp_messages



## MAIN CONSOLE
CONFIG = {
    "configurable": {
        "thread_id" : st.session_state["thread_id"]
    }
}

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Enter query: ")

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        ai_message = st.write_stream(answer_stream())

        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })