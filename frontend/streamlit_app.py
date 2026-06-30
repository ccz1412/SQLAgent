"""
Multi-Turn Text-to-SQL Agent - Streamlit Web Interface

A web-based chat interface for the Multi-Turn Text-to-SQL Agent system.
Supports both API mode (Zhipu AI) and local small model mode (Llama-3.1-8B + LoRA).

Usage:
    # Install dependencies
    pip install streamlit pandas

    # Run the app
    streamlit run frontend/streamlit_app.py

Features:
    - Chat-like interface for multi-turn dialogue
    - Real-time SQL generation and execution
    - Toggle between API mode and local model mode
    - View SQL execution results in a table
    - Conversation history sidebar
"""

import sys
import os
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Text-to-SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .sql-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "db_id" not in st.session_state:
        st.session_state.db_id = ""
    if "use_small_model" not in st.session_state:
        st.session_state.use_small_model = False
    if "dialogue_manager" not in st.session_state:
        st.session_state.dialogue_manager = None
    if "available_dbs" not in st.session_state:
        st.session_state.available_dbs = []


def load_available_databases():
    """Load available databases from Spider/BIRD datasets."""
    available_dbs = []

    # Check Spider databases
    spider_path = PROJECT_ROOT / "dat" / "spider_databases"
    if spider_path.exists():
        for db_dir in spider_path.iterdir():
            if db_dir.is_dir():
                sqlite_file = db_dir / f"{db_dir.name}.sqlite"
                if sqlite_file.exists():
                    available_dbs.append(db_dir.name)

    # Check custom test databases
    test_db_path = PROJECT_ROOT / "test_databases"
    if test_db_path.exists():
        for sqlite_file in test_db_path.glob("*.sqlite"):
            available_dbs.append(sqlite_file.stem)

    return sorted(available_dbs)


def get_dialogue_manager():
    """Get or create DialogueManager instance."""
    if st.session_state.dialogue_manager is None or \
       st.session_state.dialogue_manager.db_id != st.session_state.db_id:
        try:
            from src.dialogue.dialogue_manager import DialogueManager
            st.session_state.dialogue_manager = DialogueManager(
                db_id=st.session_state.db_id
            )
            # Override use_small_model setting
            st.session_state.dialogue_manager.use_small_model = st.session_state.use_small_model
        except Exception as e:
            st.error(f"Failed to initialize DialogueManager: {e}")
            return None
    else:
        # Update use_small_model setting
        st.session_state.dialogue_manager.use_small_model = st.session_state.use_small_model

    return st.session_state.dialogue_manager


def process_user_message(user_input: str):
    """
    Process user message through the DialogueManager.

    Args:
        user_input: User's natural language question

    Returns:
        Dict with response, sql, result, success
    """
    dm = get_dialogue_manager()
    if dm is None:
        return {
            "response": "Error: DialogueManager not initialized",
            "sql": None,
            "result": None,
            "success": False,
            "error": "DialogueManager initialization failed"
        }

    try:
        # Process the turn
        result = dm.process_turn(user_input)

        return {
            "response": result.get("response", ""),
            "sql": result.get("sql"),
            "result": result.get("result"),
            "success": result.get("success", False),
            "error": result.get("error"),
            "is_follow_up": result.get("is_follow_up", False),
            "turn_id": result.get("turn_id")
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "sql": None,
            "result": None,
            "success": False,
            "error": str(e)
        }


def display_sql(sql: str):
    """Display SQL in a formatted box."""
    st.markdown("**Generated SQL:**")
    st.code(sql, language="sql")


def display_result(result: dict):
    """Display SQL execution result as a table."""
    if result is None:
        return

    rows = result.get("rows", [])
    columns = result.get("columns", [])

    if not rows:
        st.info("Query returned no results.")
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(rows, columns=columns)
    st.markdown(f"**Query Result** ({len(rows)} rows):")
    st.dataframe(df, use_container_width=True)


def display_chat_message(role: str, content: str, sql: str = None, result: dict = None, error: str = None):
    """Display a chat message with optional SQL and result."""
    with st.chat_message(role):
        st.markdown(content)

        if sql:
            display_sql(sql)

        if result:
            display_result(result)

        if error:
            st.markdown(f'<div class="error-box">⚠️ <b>Error:</b> {error}</div>',
                        unsafe_allow_html=True)


def sidebar_settings():
    """Render sidebar settings."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Database selection
        st.subheader("Database")
        available_dbs = load_available_databases()
        if available_dbs:
            selected_db = st.selectbox(
                "Select database",
                options=available_dbs,
                index=0 if available_dbs else None,
                help="Choose the database to query"
            )
            st.session_state.db_id = selected_db
        else:
            st.warning("No databases found. Please check `dat/spider_databases/` or `test_databases/`.")
            custom_db = st.text_input("Or enter custom db_id:", key="custom_db")
            st.session_state.db_id = custom_db

        # Model selection
        st.subheader("Model")
        model_mode = st.radio(
            "Select model mode",
            options=["API (Zhipu AI)", "Local Small Model (Llama-3.1-8B)"],
            index=0,
            help="API mode uses Zhipu AI (glm-4-flash). Local mode uses your fine-tuned Llama-3.1-8B."
        )
        st.session_state.use_small_model = (model_mode == "Local Small Model (Llama-3.1-8B)")

        if st.session_state.use_small_model:
            st.info("💡 Local model mode selected.\n\nMake sure you have:\n- PyTorch + Transformers installed\n- Enough GPU memory (~4-5GB for 4-bit quantized model)")

        # Reset conversation
        st.subheader("Conversation")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.dialogue_manager = None
            st.success("Conversation reset!")
            st.rerun()

        # Show conversation history
        if st.session_state.messages:
            st.subheader("History")
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.text(f"{i//2 + 1}. {msg['content'][:30]}...")


def main():
    """Main Streamlit app."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">🤖 Text-to-SQL Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Turn Natural Language Database Query System</div>', unsafe_allow_html=True)

    # Sidebar
    sidebar_settings()

    # Check if database is selected
    if not st.session_state.db_id:
        st.warning("⚠️ Please select a database in the sidebar to start.")
        return

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sql=message.get("sql"),
            result=message.get("result"),
            error=message.get("error")
        )

    # Chat input
    user_input = st.chat_input("Ask a question about the database...")

    if user_input:
        # Display user message
        display_chat_message("user", user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Process with Agent
        with st.spinner("🤔 Agent is thinking..."):
            result = process_user_message(user_input)

        # Display assistant response
        if result["success"]:
            response_text = result["response"]
            if result.get("is_follow_up"):
                response_text = f"🔄 (Follow-up detected)\n\n{response_text}"

            display_chat_message(
                "assistant",
                response_text,
                sql=result.get("sql"),
                result=result.get("result")
            )

            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sql": result.get("sql"),
                "result": result.get("result")
            })
        else:
            error_msg = result.get("error", "Unknown error")
            display_chat_message(
                "assistant",
                f"Sorry, I encountered an error.",
                error=error_msg
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error.",
                "error": error_msg
            })

        # Rerun to update UI
        st.rerun()


if __name__ == "__main__":
    main()
