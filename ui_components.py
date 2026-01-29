"""
UI Components Module
Contains reusable Streamlit UI components
"""

import streamlit as st
import os
from typing import Dict, List, Tuple, Any

from config import UI_TEXT, TOP_K
from utils import check_database_exists, get_db_path


def render_header():
    """Render application header"""
    from config import APP_TITLE, APP_DESCRIPTION
    st.title(f"🔍 {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)


def render_sidebar(embedding_models: Dict, base_dir: str) -> Dict[str, Any]:
    """
    Render sidebar with configuration options
    
    Args:
        embedding_models: Dictionary of available models
        base_dir: Base directory for vector databases
    
    Returns:
        Dictionary with selected configuration
    """
    with st.sidebar:
        st.header(UI_TEXT["sidebar_header"])
        
        # Operation Mode Selection
        from config import OPERATION_MODES
        st.subheader(UI_TEXT["operation_mode_header"])
        operation_mode = st.radio(
            "Select operation mode:",
            options=list(OPERATION_MODES.keys()),
            format_func=lambda x: OPERATION_MODES[x],
            index=0
        )
        
        if operation_mode == "retrieval_only":
            st.info("ℹ️ Documents will be retrieved without AI-generated answers")
        
        # Model selection
        st.subheader(UI_TEXT["model_selection_header"])
        selected_model = st.selectbox(
            "Choose an embedding model:",
            options=list(embedding_models.keys()),
            index=0
        )
        
        model_id, folder_name = embedding_models[selected_model]
        db_path = get_db_path(base_dir, folder_name)
        
        # Show model details
        with st.expander("Model Details"):
            st.code(
                f"Model ID: {model_id}\n"
                f"Folder: {folder_name}\n"
                f"Path: {db_path}"
            )
        
        # --- PERUBAHAN: Input API Key DIHAPUS ---
        # --- PERUBAHAN: Advanced Settings (Distance) DIHAPUS ---
        
        # Database status
        st.subheader(UI_TEXT["db_status_header"])
        _render_database_status(db_path)
        
        st.divider()
        st.caption(f"💡 Top-K retrieval: {TOP_K} documents")
    
    return {
        'selected_model': selected_model,
        'model_id': model_id,
        'folder_name': folder_name,
        'db_path': db_path,
        'operation_mode': operation_mode
        # 'api_key' dan 'distance_threshold' tidak lagi dikembalikan dari UI
    }


def _render_database_status(db_path: str):
    """
    Render database status indicator
    
    Args:
        db_path: Path to database directory
    """
    from config import FAISS_INDEX_FILENAME, PICKLE_FILENAME
    
    if os.path.exists(db_path):
        faiss_exists = os.path.exists(os.path.join(db_path, FAISS_INDEX_FILENAME))
        pkl_exists = os.path.exists(os.path.join(db_path, PICKLE_FILENAME))
        
        if faiss_exists and pkl_exists:
            st.success("✅ Database files found")
        else:
            st.error("❌ Missing index.faiss or index.pkl")
    else:
        st.warning(f"⚠️ Database folder not found")
        st.info("Run `setup_folders.py` to create the structure")


def render_query_input() -> str:
    """
    Render query input area
    
    Returns:
        User query string
    """
    st.subheader("📝 Enter Your Query")
    query = st.text_area(
        "Type your question here:",
        height=150,
        placeholder=UI_TEXT["query_placeholder"]
    )
    return query


def render_results(
    retrieved_docs: List[Tuple[str, float]],
    answer: str,
    context_container,
    model_id: str,
    query: str,
    operation_mode: str = "retrieval_generation"
):
    """
    Render retrieval results and generated answer
    
    Args:
        retrieved_docs: List of (document, distance) tuples
        answer: Generated answer text (empty if retrieval only)
        context_container: Streamlit container for context display
        model_id: Embedding model identifier
        query: Original user query
        operation_mode: Current operation mode
    """
    # Display retrieved context
    with context_container:
        st.info(f"📊 Retrieved {len(retrieved_docs)} most relevant documents")
        for i, (doc, distance) in enumerate(retrieved_docs):
            with st.expander(f"📄 Document {i+1} (Distance: {distance:.4f})", expanded=(i==0)):
                st.write(doc)
    
    # Display generated answer only in generation mode
    if operation_mode == "retrieval_generation" and answer:
        st.divider()
        st.subheader("💬 Generated Answer")
        st.markdown(answer)
        
        # Show metadata
        with st.expander("ℹ️ Generation Info"):
            from config import GEMINI_MODEL
            st.json({
                "operation_mode": "Retrieval + Generation",
                "embedding_model": model_id,
                "llm_model": GEMINI_MODEL,
                "retrieved_docs": len(retrieved_docs),
                "query_length": len(query),
                "top_k": TOP_K
            })
    else:
        # Retrieval only mode
        st.divider()
        st.success("✅ Retrieval Complete - Documents retrieved successfully")
        
        # Show metadata
        with st.expander("ℹ️ Retrieval Info"):
            st.json({
                "operation_mode": "Retrieval Only",
                "embedding_model": model_id,
                "retrieved_docs": len(retrieved_docs),
                "query_length": len(query),
                "top_k": TOP_K
            })


def render_error(title: str, message: str, help_text: str = None):
    """
    Render error message with optional help text
    """
    st.error(f"❌ {title}")
    st.warning(message)
    if help_text:
        st.info(f"💡 {help_text}")


def render_success(message: str):
    st.success(f"✅ {message}")


def render_info(message: str):
    st.info(f"ℹ️ {message}")