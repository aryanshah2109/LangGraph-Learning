import streamlit as st
from backend import chatbot
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


def create_message_history(messages):
    
    temp_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role="user"
        else:
            role="assistant"
        temp_messages.append({
            "role": role,
            "content": message.content
        })

    return temp_messages



def get_chat_name(thread_id):

    message_history = create_message_history(get_messages(thread_id))

    prompt = f"""Create a short name to label the chat for this message history similar to how ChatGPT names its chats:
    {message_history[-2:]}

    
    """
    response = chatbot.invoke({
            "messages": [HumanMessage(prompt)]
        }, 
        config = {
            "configurable" : {
                "thread_id": thread_id
            }
        }
    )["messages"][-1].content

    print(response)

    return response

## SESSION MANAGEMENT

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads" not in st.session_state:
    st.session_state["threads"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Add main first thread in session state threads
add_thread(st.session_state["thread_id"])


## SIDEBAR UI
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My chats")


for thread_id in st.session_state["threads"][::-1]:

    if f"btn_{thread_id}" not in st.session_state:
        st.session_state[f"btn_{thread_id}"] = str(thread_id)

    else:
        st.session_state[f"btn_{thread_id}"] = get_chat_name(thread_id)

    if st.sidebar.button(st.session_state[f"btn_{thread_id}"]):
        st.session_state["thread_id"] = thread_id

        # Get past messages from RAM (DB if in prod)
        messages = get_messages(thread_id)
        
        st.session_state["message_history"] = create_message_history(messages=messages)
        


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