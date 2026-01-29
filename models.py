"""
Embedding Models Module
Handles loading and managing sentence transformer models
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict


class EmbeddingModelManager:
    """
    Manager class for embedding models
    Handles loading, caching, and reusing models
    """
    
    def __init__(self):
        """Initialize the model manager"""
        self.loaded_models: Dict[str, SentenceTransformer] = {}
    
    def _load_single_model(self, model_id: str) -> Optional[SentenceTransformer]:
        """
        Load a single sentence transformer model
        
        Args:
            model_id: Hugging Face model identifier
        
        Returns:
            Loaded SentenceTransformer model or None if failed
        """
        try:
            model = SentenceTransformer(model_id)
            return model
        except Exception as e:
            st.error(f"Error loading model {model_id}: {str(e)}")
            return None
    
    def preload_all_models(self, model_configs: Dict) -> Dict[str, bool]:
        """
        Preload all available models at startup
        
        Args:
            model_configs: Dictionary of model configurations
        
        Returns:
            Dictionary of model_id: success status
        """
        results = {}
        total_models = len(model_configs)
        
        st.info(f"🔄 Preloading {total_models} embedding models...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (display_name, (model_id, _)) in enumerate(model_configs.items(), 1):
            status_text.text(f"Loading {idx}/{total_models}: {display_name}")
            
            if model_id not in self.loaded_models:
                model = self._load_single_model(model_id)
                if model:
                    self.loaded_models[model_id] = model
                    results[model_id] = True
                    st.success(f"✅ Loaded: {display_name}")
                else:
                    results[model_id] = False
                    st.warning(f"⚠️ Failed: {display_name}")
            else:
                results[model_id] = True
                st.info(f"ℹ️ Already loaded: {display_name}")
            
            progress_bar.progress(idx / total_models)
        
        status_text.empty()
        progress_bar.empty()
        
        successful = sum(1 for v in results.values() if v)
        st.success(f"✅ Preloaded {successful}/{total_models} models successfully!")
        
        return results
    
    def load_model(self, model_id: str) -> Optional[SentenceTransformer]:
        """
        Load or retrieve cached model
        
        Args:
            model_id: Hugging Face model identifier
        
        Returns:
            Loaded SentenceTransformer model or None
        """
        # Return from cache if already loaded
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # Load new model if not in cache
        with st.spinner(f"Loading model: {model_id}..."):
            model = self._load_single_model(model_id)
            if model:
                self.loaded_models[model_id] = model
            return model
    
    def get_loaded_model(self, model_id: str) -> Optional[SentenceTransformer]:
        """
        Get already loaded model without triggering new load
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model if loaded, None otherwise
        """
        return self.loaded_models.get(model_id)
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a specific model is loaded
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if model is loaded, False otherwise
        """
        return model_id in self.loaded_models
    
    def get_loaded_models_count(self) -> int:
        """
        Get count of loaded models
        
        Returns:
            Number of loaded models
        """
        return len(self.loaded_models)
    
    def clear_cache(self):
        """Clear all loaded models from memory"""
        self.loaded_models.clear()