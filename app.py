import os

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from rag_agent import invoke_our_graph, update_retriever
from st_callable_util import get_retriever_cb, get_streamlit_cb

# Load the project config file
with open("configs.yaml", "r", encoding="utf8") as file:
    project_configs = yaml.safe_load(file)

load_dotenv()

st.markdown(
    """
    <h1 style="text-align: center;">RAG with PDF Highlights</h1>
    """,
    unsafe_allow_html=True
)

folder_path = project_configs["vectordb-folder-path"]
vdb_names = [
    d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
]


def initialize_new_chat():
    """clear messages"""
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]
    update_retriever()


if "messages" not in st.session_state:
    # default initial message to render in message state
    initialize_new_chat()

st.sidebar.title("Knowledge Base")
dropdown_selection = st.sidebar.selectbox(
    "📃 Select a PDF",
    key="user_input_db_name",
    options=vdb_names,
    index=0,  # Default selected option
    on_change=initialize_new_chat,
)

# call the retriever function
update_retriever()

if len(st.session_state["messages"]) == 0 or st.sidebar.button("Clear Chat 🧹"):
    initialize_new_chat()

for msg in st.session_state["messages"]:
    st.chat_message(msg.type).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything about the PDF!"):
    st.chat_message("user").write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        retrieval_handler = get_retriever_cb(st.container())
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(
            st.session_state.messages[-1], [st_callback, retrieval_handler]
        )
        st.session_state.messages.append(
            AIMessage(content=response["messages"][-1].content)
        )
