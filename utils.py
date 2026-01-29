"""
Utility Functions Module
Contains helper functions used across the application
"""

import os
from pathlib import Path
from typing import List

from config import FAISS_INDEX_FILENAME, PICKLE_FILENAME


def sanitize_model_name(model_name: str) -> str:
    """
    Convert model name to folder-safe format
    
    Args:
        model_name: Original model name (e.g., "org/model-name")
    
    Returns:
        Sanitized folder name (e.g., "org_model-name")
    """
    return model_name.replace("/", "_")


def get_db_path(base_dir: str, folder_name: str) -> str:
    """
    Construct database path from base directory and folder name
    
    Args:
        base_dir: Base directory for all databases
        folder_name: Specific model folder name
    
    Returns:
        Full path to database directory
    """
    return os.path.join(base_dir, f"db-{folder_name}")


def check_database_exists(db_path: str) -> bool:
    """
    Check if database directory and required files exist
    
    Args:
        db_path: Path to database directory
    
    Returns:
        True if all required files exist, False otherwise
    """
    if not os.path.exists(db_path):
        return False
    
    faiss_path = os.path.join(db_path, FAISS_INDEX_FILENAME)
    pkl_path = os.path.join(db_path, PICKLE_FILENAME)
    
    return os.path.exists(faiss_path) and os.path.exists(pkl_path)


def create_db_directory(db_path: str) -> bool:
    """
    Create database directory if it doesn't exist
    
    Args:
        db_path: Path to create
    
    Returns:
        True if created successfully, False otherwise
    """
    try:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False


def get_all_db_paths(base_dir: str, model_mapping: dict) -> List[str]:
    """
    Get all database paths for given models
    
    Args:
        base_dir: Base directory for databases
        model_mapping: Dictionary of model configurations
    
    Returns:
        List of database paths
    """
    paths = []
    for _, folder_name in model_mapping.values():
        paths.append(get_db_path(base_dir, folder_name))
    return paths


def format_distance(distance: float) -> str:
    """
    Format distance score for display
    
    Args:
        distance: Raw distance value
    
    Returns:
        Formatted string
    """
    return f"{distance:.4f}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."