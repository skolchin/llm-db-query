# Supplementary Chat UI for LangChain AI agent.
#
# Run it with streamlit:
#
#     streamlit run lng/app.py --server.address 127.0.0.1 --server.port 7932
#
import os
import json
import altair as alt
import plotly.graph_objects as go
import streamlit as st
from uuid import uuid4
from datetime import datetime
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    messages_to_dict,
    messages_from_dict
)
from langchain_core.runnables.config import RunnableConfig
from boring_semantic_layer.agents.backends.langgraph import LangGraphBackend
from typing import Generator, List, Dict, Any

# Uncomment to use sql agent
# from sql_agent import app, SYSTEM_PROMPT, MODEL_TYPE

# Uncomment to use BSL agent
from bsl_agent import app, SYSTEM_PROMPT, MODEL_TYPE

# Constants
HISTORY_DIR = ".history"
CONVERSATION_FILE = "conversation_{id}.json"

os.makedirs(HISTORY_DIR, exist_ok=True)

# Display roles support
DISPLAY_ROLES_MAP = {
    'human': 'user', 
    'ai': 'assistant',
}
def display_role(msg: BaseMessage) -> str | None:
    """ Find out the label of message onscreen """

    if isinstance(msg, AIMessage) and msg.tool_calls:
        return DISPLAY_ROLES_MAP.get('tool')
    
    return DISPLAY_ROLES_MAP.get(msg.type)

# Initial messages list
def get_initial_messages_list() -> List[BaseMessage]:
    """ Build a messages list for a new conversation """
    return [SystemMessage(SYSTEM_PROMPT)]

# Conversation support
def get_conversation_file_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(HISTORY_DIR, CONVERSATION_FILE.format(id=conversation_id))

def save_conversation(conversation_id: str, messages: List[BaseMessage]) -> None:
    """Save conversation messages to JSON file."""
    file_path = get_conversation_file_path(conversation_id)
    serialized = messages_to_dict(messages)
    with open(file_path, "wt") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

def load_conversation(conversation_id: str) -> List[BaseMessage]:
    """Load conversation messages from JSON file."""
    file_path = get_conversation_file_path(conversation_id)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            return messages_from_dict(data)
    return []

def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations from history directory."""
    conversations = []
    if not os.path.exists(HISTORY_DIR):
        return []
    
    for filename in os.listdir(HISTORY_DIR):
        if filename.startswith("conversation_") and filename.endswith(".json"):
            conversation_id = filename[len("conversation_"):-len(".json")]
            file_path = os.path.join(HISTORY_DIR, filename)
            mtime = os.path.getmtime(file_path)

            # Load first user message as title preview
            messages = load_conversation(conversation_id)
            title = f"Conversation {conversation_id[:8]}"
            for msg in messages:
                if msg.type == 'human':
                    content = str(msg.content)
                    title = content[:50] + ("..." if len(content) > 50 else "")
                    break

            conversations.append({
                "id": conversation_id,
                "title": title,
                "updated_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            })
    # Sort by most recently updated
    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations

def create_new_conversation() -> str:
    """Create a new conversation ID."""
    return str(uuid4())


def get_sql_agent_response(messages: list, conversation_id: str) -> Generator[str | alt.Chart | go.Figure, None, None]:
    """Get sql agent (actually workflow) response with streaming support."""

    config = RunnableConfig({"configurable": {"thread_id": conversation_id}})

    assert isinstance(app, CompiledStateGraph)

    for chunk in app.stream(
        {"messages": messages},
        stream_mode="values",
        config=config
    ):
        message = chunk["messages"][-1]
        if display_role(message) == 'assistant':
            content = str(message.content)
            yield content

def get_bsl_agent_response(messages: list, conversation_id: str) -> Generator[str | alt.Chart | go.Figure, None, None]:
    """Get BSL agent response with streaming support """

    # Get the last human message from the messages list
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = str(msg.content)
            break
    
    if not last_user_message:
        return

    assert isinstance(app, LangGraphBackend)

    # Callbacks
    tool_output = ""
    final_response = ""
    thinking_buffer = []
    
    def on_thinking(text):
        thinking_buffer.append(text)
    
    def on_tool_call(name, args, tokens):
        pass
    
    def on_tool_result(tool_call_id, status, error, content):
        nonlocal tool_output
        if content:
            tool_output = content
    
    # Call the BSL agent query method
    try:
        tool_output, final_response = app.query(
            last_user_message,
            on_tool_call=on_tool_call,
            on_thinking=on_thinking,
            on_tool_result=on_tool_result
        )
        
        # Yield any thinking
        for thought in thinking_buffer:
            yield thought
        
        # Yield the response
        if final_response:
            yield final_response

        # Check tool output contains a chart definition, if yes - try to recover and yield result
        if tool_output and '{"total_rows"' in tool_output:
            try:
                js = json.loads(tool_output[tool_output.index('{"total_rows"'):])
                match js['chart']['backend']:
                    case 'altair':
                        yield alt.Chart.from_dict(js['chart']['data'])
                    case 'plotly':
                        yield go.Figure(js['chart']['data'])
                        
            except (json.JSONDecodeError, KeyError) as ex:
                print(f"Error {type(ex)} retrieving chart: {ex}")
            
    except Exception as e:
        yield f"Error: {str(e)}"

def get_agent_response(messages: list, conversation_id: str) -> Generator[str | alt.Chart | go.Figure, None, None]:
    match app:
        case CompiledStateGraph():
            return get_sql_agent_response(messages, conversation_id)
        case LangGraphBackend():
            return get_bsl_agent_response(messages, conversation_id)
        case _:
            raise ValueError('Uknown app type')

# Streamlit page config
st.set_page_config(page_title="SQL Database Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .conversation-item {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        cursor: pointer;
    }
    .conversation-item:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = create_new_conversation()
if "messages" not in st.session_state:
    # Initialize with system prompt
    st.session_state.messages = get_initial_messages_list()
if "conversations" not in st.session_state:
    st.session_state.conversations = list_conversations()

# Sidebar - Conversation list
with st.sidebar:
    st.title("💬 Conversations")
    
    # New conversation button
    if st.button("➕ New Conversation"):
        st.session_state.conversation_id = create_new_conversation()
        st.session_state.messages = get_initial_messages_list()
        st.session_state.conversations = list_conversations()
        st.rerun()
    
    st.divider()
    
    # Conversation list
    st.subheader("History")
    
    if not st.session_state.conversations:
        st.info("No conversations yet. Start a new one!")
    else:
        for conv in st.session_state.conversations:
            if st.button(
                f"📝 {conv['title']}\n🕐 {conv['updated_at']}",
                key=f"conv_{conv['id']}",
                help=conv['id']
            ):
                # Save current conversation first
                if len(st.session_state.messages) > 1:
                    save_conversation(st.session_state.conversation_id, st.session_state.messages)
                
                # Load selected conversation
                st.session_state.conversation_id = conv['id']
                st.session_state.messages = load_conversation(conv['id'])
                st.rerun()
        
        st.divider()
        
        # Clear all conversations button
        if st.button("🗑️ Clear All History"):
            for conv in st.session_state.conversations:
                file_path = get_conversation_file_path(conv['id'])
                if os.path.exists(file_path):
                    os.remove(file_path)
            st.session_state.conversations = []
            st.session_state.conversation_id = create_new_conversation()
            st.session_state.messages = get_initial_messages_list()
            st.rerun()

# Main chat area
st.title("🤖 SQL Database Chatbot")

# Display chat messages
for msg in st.session_state.messages:
    if role := display_role(msg):
        content = str(msg.content)
        with st.chat_message(role):
            st.markdown(content)

# Chat input
if query := st.chat_input(f"Ask a question ({MODEL_TYPE.capitalize()})..."):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add to messages
    st.session_state.messages.append(HumanMessage(query))
    
    # Get assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(':gray[*Thinking...*]')
        full_response = ""
        
        response_generator = get_agent_response(st.session_state.messages, st.session_state.conversation_id)
        try:
            for chunk in response_generator:
                match chunk:
                    case str():
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                    case alt.Chart():
                        st.altair_chart(chunk, width='content')

                    case go.Figure():
                        st.plotly_chart(chunk, width='content')

        except Exception as e:
            full_response = f"Error: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to messages for display
    if full_response:
        st.session_state.messages.append(AIMessage(full_response))
    
    # Save conversation
    save_conversation(st.session_state.conversation_id, st.session_state.messages)
    st.session_state.conversations = list_conversations()
