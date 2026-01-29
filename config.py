"""
Configuration module
Contains all application settings and constants
"""

# Application metadata
APP_TITLE = "RAG Application with FAISS & Gemini"
APP_ICON = "🔍"
APP_DESCRIPTION = "Retrieval-Augmented Generation using custom embedding models and Google Gemini"

# Page configuration for Streamlit
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide"
}

# --- PERUBAHAN 1: HARDCODE API KEY DI SINI ---
# Masukkan API Key Gemini Anda di dalam tanda kutip

EMBEDDING_MODELS = {
    "Indobert Base (Trash Small)": (
        "yosriku/Indobert-Base-p2-Trash-Small-EXP3",
        "yosriku_Indobert-Base-p2-Trash-Small-EXP3"
    ),
    "NusaBERT Large v4": (
        "LazarusNLP/all-nusabert-large-v4",
        "LazarusNLP_all-nusabert-large-v4"
    ),
    "All MiniLM L6 v2": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers_all-MiniLM-L6-v2"
    )
}

# Directory settings
BASE_DIR = "./vector_dbs"
FAISS_INDEX_FILENAME = "index.faiss"
PICKLE_FILENAME = "index.pkl"

TOP_K = 3

# Default distance threshold (karena UI dihapus)
# Set None untuk mengambil semua hasil top-k tanpa batasan jarak
DISTANCE_THRESHOLD = None 

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://makersuite.google.com/app/apikey"

# Operation modes
OPERATION_MODES = {
    "retrieval_generation": "🤖 Retrieval + Generation (RAG)",
    "retrieval_only": "🔍 Retrieval Only"
}

# UI text constants
UI_TEXT = {
    "sidebar_header": "⚙️ Configuration",
    "operation_mode_header": "1. Operation Mode",
    "model_selection_header": "2. Select Embedding Model",
    # Header API Key dihapus
    "db_status_header": "3. Database Status",
    "query_placeholder": "e.g., Apa topik utama yang dibahas dalam dokumen?",
    "retrieval_only_note": "📋 Retrieval Only mode - No answer generation"

}
