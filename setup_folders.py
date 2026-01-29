"""
Setup script to create the required folder structure for RAG application.
This script creates all necessary directories for each embedding model.

Usage:
    python setup_folders.py
"""

import os
from pathlib import Path
from config import EMBEDDING_MODELS, BASE_DIR
from utils import get_db_path, create_db_directory


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("RAG Application - Database Folder Setup")
    print("=" * 60)
    print()


def create_all_folders() -> list:
    """
    Create folder structure for all models
    
    Returns:
        List of created folder paths
    """
    # Create base directory
    base_path = Path(BASE_DIR)
    base_path.mkdir(exist_ok=True)
    print(f"✅ Created base directory: {BASE_DIR}")
    print()
    
    # Create subdirectories for each model
    created_folders = []
    
    for model_display, (model_id, folder_name) in EMBEDDING_MODELS.items():
        db_path = get_db_path(BASE_DIR, folder_name)
        
        if create_db_directory(db_path):
            created_folders.append({
                'path': db_path,
                'model_id': model_id,
                'display': model_display
            })
            print(f"📁 Created: {db_path}")
            print(f"   Model: {model_id}")
            print(f"   Display: {model_display}")
            print()
        else:
            print(f"❌ Failed to create: {db_path}")
            print()
    
    return created_folders


def print_instructions(created_folders: list):
    """
    Print usage instructions
    
    Args:
        created_folders: List of created folder information
    """
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("-" * 60)
    print("For each model you want to use, place the following files:")
    print()
    
    for i, folder_info in enumerate(created_folders, 1):
        print(f"{i}. In folder: {folder_info['path']}")
        print(f"   Model: {folder_info['display']}")
        print(f"   Required files:")
        print(f"   - index.faiss  (FAISS IndexFlatL2 vector database)")
        print(f"   - index.pkl    (Pickle file with document chunks)")
        print()
    
    print("=" * 60)
    print("File Requirements:")
    print("=" * 60)
    print("• index.faiss: FAISS index created with IndexFlatL2")
    print("• index.pkl:   Python list of text chunks (use pickle.dump)")
    print()
    print("Example to create index.pkl:")
    print("-" * 60)
    print("import pickle")
    print("documents = ['chunk 1', 'chunk 2', 'chunk 3', ...]")
    print("with open('index.pkl', 'wb') as f:")
    print("    pickle.dump(documents, f)")
    print()
    print("=" * 60)
    print("After placing your files, run: streamlit run app.py")
    print("=" * 60)


def verify_structure():
    """Verify the created folder structure"""
    from config import FAISS_INDEX_FILENAME, PICKLE_FILENAME
    from utils import check_database_exists
    
    print()
    print("=" * 60)
    print("Verification Report")
    print("=" * 60)
    print()
    
    base_path = Path(BASE_DIR)
    
    if not base_path.exists():
        print("❌ Base directory does not exist!")
        return False
    
    all_good = True
    
    for model_display, (model_id, folder_name) in EMBEDDING_MODELS.items():
        db_path = get_db_path(BASE_DIR, folder_name)
        print(f"📂 {os.path.basename(db_path)}")
        
        if os.path.exists(db_path):
            print(f"   ✅ Folder exists")
            
            # Check for required files
            faiss_file = os.path.join(db_path, FAISS_INDEX_FILENAME)
            pkl_file = os.path.join(db_path, PICKLE_FILENAME)
            
            if os.path.exists(faiss_file):
                print(f"   ✅ {FAISS_INDEX_FILENAME} found")
            else:
                print(f"   ⚠️  {FAISS_INDEX_FILENAME} MISSING")
                all_good = False
            
            if os.path.exists(pkl_file):
                print(f"   ✅ {PICKLE_FILENAME} found")
            else:
                print(f"   ⚠️  {PICKLE_FILENAME} MISSING")
                all_good = False
        else:
            print(f"   ❌ Folder does not exist")
            all_good = False
        
        print()
    
    if all_good:
        print("✅ All folders and files are in place!")
    else:
        print("⚠️  Some files are missing. Please add them before running the app.")
    
    print("=" * 60)
    
    return all_good


def main():
    """Main setup execution"""
    try:
        print_header()
        created_folders = create_all_folders()
        print_instructions(created_folders)
        verify_structure()
        
        print()
        print("🎉 Setup script completed successfully!")
        print()
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()