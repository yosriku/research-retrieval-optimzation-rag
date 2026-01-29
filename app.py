"""
Main Streamlit Application Entry Point
Handles UI and orchestrates the RAG pipeline
"""

import streamlit as st
import os
from pathlib import Path

# --- PERUBAHAN: Import GEMINI_API_KEY dan DISTANCE_THRESHOLD dari config ---
from config import EMBEDDING_MODELS, BASE_DIR, TOP_K, PAGE_CONFIG, GEMINI_API_KEY, DISTANCE_THRESHOLD
from models import EmbeddingModelManager
from retrieval import FAISSRetriever
from generation import GeminiGenerator
from ui_components import (
    render_sidebar,
    render_header,
    render_query_input,
    render_results,
    render_error
)
from utils import check_database_exists, get_db_path

# Page configuration
st.set_page_config(**PAGE_CONFIG)


def initialize_session_state():
    """Initialize session state variables"""
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingModelManager()
    if 'models_preloaded' not in st.session_state:
        st.session_state.models_preloaded = False
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'current_model_id' not in st.session_state:
        st.session_state.current_model_id = None
    if 'current_db_path' not in st.session_state:
        st.session_state.current_db_path = None


def load_components(model_id, db_path, api_key):
    """
    Load or reload components if configuration changed
    """
    # Get embedding model from preloaded cache
    embedding_model = st.session_state.embedding_manager.get_loaded_model(model_id)
    
    if embedding_model is None:
        st.error(f"❌ Model {model_id} not loaded. Please restart the application.")
        raise Exception("Embedding model not available")
    
    # Load retriever (reload if database path changed)
    if (st.session_state.retriever is None or 
        st.session_state.current_db_path != db_path):
        
        with st.spinner("Loading vector database..."):
            st.session_state.retriever = FAISSRetriever(db_path)
            st.session_state.current_db_path = db_path
        
        st.success(f"✅ Loaded {st.session_state.retriever.get_index_size()} vectors")
    
    # Initialize generator (only if API key is provided)
    generator = None
    if api_key:
        generator = GeminiGenerator(api_key)
    
    return embedding_model, st.session_state.retriever, generator


def process_query(query, embedding_model, retriever, generator, operation_mode, distance_threshold):
    """
    Process user query through RAG pipeline
    """
    # Retrieval step (always performed)
    with st.spinner("🔍 Retrieving relevant documents..."):
        retrieved_docs = retriever.retrieve(
            query, 
            embedding_model, 
            top_k=TOP_K,
            distance_threshold=distance_threshold
        )
    
    # Check if any documents were retrieved
    if not retrieved_docs:
        st.warning(f"⚠️ No documents found matching the query.")
        return [], ""
    
    # Generation step (only if mode is retrieval_generation)
    answer = ""
    if operation_mode == "retrieval_generation" and generator:
        with st.spinner("💬 Generating answer with Gemini..."):
            answer = generator.generate(query, retrieved_docs)
    
    return retrieved_docs, answer


def main():
    """Main application logic"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Preload all models on first run
    if not st.session_state.models_preloaded:
        with st.spinner("🔄 Initializing application..."):
            st.info("📦 Downloading and loading all embedding models. This may take a few minutes on first run...")
            results = st.session_state.embedding_manager.preload_all_models(EMBEDDING_MODELS)
            st.session_state.models_preloaded = True
            
            # Show summary
            failed_models = [model_id for model_id, success in results.items() if not success]
            if failed_models:
                st.warning(f"⚠️ Some models failed to load: {', '.join(failed_models)}")
            else:
                st.success("🎉 All models loaded successfully!")
            
            st.info("💡 Models are now cached. Subsequent runs will be faster.")
    
    # Render sidebar and get configuration
    # config yang dikembalikan UI tidak lagi memiliki 'api_key' atau 'distance_threshold'
    config = render_sidebar(EMBEDDING_MODELS, BASE_DIR)
    
    selected_model_display = config['selected_model']
    operation_mode = config['operation_mode']
    
    # --- PERBAIKAN UTAMA: Ambil API Key & Threshold langsung dari variabel config.py ---
    api_key = GEMINI_API_KEY 
    distance_threshold = DISTANCE_THRESHOLD
    
    model_id, folder_name = EMBEDDING_MODELS[selected_model_display]
    db_path = get_db_path(BASE_DIR, folder_name)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        query = render_query_input()
        
        # Dynamic button text based on mode
        button_text = "🔎 Search & Generate Answer" if operation_mode == "retrieval_generation" else "🔍 Search Documents"
        search_button = st.button(button_text, type="primary", use_container_width=True)
        
        # Show mode indicator
        if operation_mode == "retrieval_only":
            st.info("📋 Retrieval Only mode - Documents will be retrieved without AI answer generation")
    
    with col2:
        st.subheader("📊 Retrieved Context")
        context_container = st.container()
    
    # Process query when button clicked
    if search_button:
        # Validation
        if not query.strip():
            st.warning("⚠️ Please enter a query")
            return
        
        # Validasi API Key hardcoded
        if operation_mode == "retrieval_generation" and (not api_key or api_key == "MASUKKAN_API_KEY_GEMINI_ANDA_DISINI"):
            st.error("⚠️ API Key belum dikonfigurasi. Silakan edit file `config.py` dan masukkan API Key Gemini Anda.")
            return
        
        # Check if database exists
        if not check_database_exists(db_path):
            render_error(
                "Database files not found",
                f"Missing files in: {db_path}",
                "Run setup_folders.py and add your database files"
            )
            return
        
        try:
            # Load all components
            embedding_model, retriever, generator = load_components(
                model_id, db_path, api_key if operation_mode == "retrieval_generation" else None
            )
            
            # Process query through RAG pipeline
            retrieved_docs, answer = process_query(
                query, embedding_model, retriever, generator, operation_mode, distance_threshold
            )
            
            # Render results
            render_results(
                retrieved_docs=retrieved_docs,
                answer=answer,
                context_container=context_container,
                model_id=model_id,
                query=query,
                operation_mode=operation_mode
            )
        
        except FileNotFoundError as e:
            render_error(
                "File not found",
                str(e),
                "Ensure database files are in the correct location"
            )
        
        except Exception as e:
            render_error("An error occurred", str(e))
            st.exception(e)
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit 🎈 | Powered by FAISS & Google Gemini ✨")


if __name__ == "__main__":
    main()