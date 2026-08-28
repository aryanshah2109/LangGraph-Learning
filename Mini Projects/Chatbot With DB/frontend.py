import streamlit as st
from backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

## UTILITY FUNCTIONS
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def add_thread(thread_id):
    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].append(thread_id)

def reset_chat():
    
    new_thread_id = generate_thread_id()
    add_thread(st.session_state["thread_id"])
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []

def get_messages(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

## SESSION MANAGEMENT

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads" not in st.session_state:
    st.session_state["threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Add main first thread in session state threads
add_thread(st.session_state["thread_id"])


## SIDEBAR UI
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My chats")


for thread_id in st.session_state['threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = get_messages(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages     


CONFIG = {
    'configurable': {
        'thread_id': st.session_state['thread_id']
    }
}

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})