"""
Generation Module
Handles text generation using Google Gemini API
"""

import google.generativeai as genai
from typing import List, Tuple

from config import GEMINI_MODEL


class GeminiGenerator:
    """
    Generator class for Google Gemini API
    Handles prompt construction and answer generation
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini generator with API key
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        self._configure_api()
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def _configure_api(self):
        """Configure Gemini API with provided key"""
        genai.configure(api_key=self.api_key)
    
    def _build_prompt(
        self, 
        query: str, 
        context_docs: List[Tuple[str, float]]
    ) -> str:
        """
        Build prompt with context and query
        
        Args:
            query: User query
            context_docs: List of (document_text, distance) tuples
        
        Returns:
            Formatted prompt string
        """
        # Format context documents
        context_parts = []
        for i, (doc, distance) in enumerate(context_docs):
            context_parts.append(f"[Document {i+1}]:\n{doc}")
        
        context_text = "\n\n".join(context_parts)
        
        # Construct full prompt
        prompt = f"""Based on the following context as human as possible, please answer the question in Indonesia. If the answer cannot be found in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(
        self, 
        query: str, 
        context_docs: List[Tuple[str, float]]
    ) -> str:
        """
        Generate answer using Gemini API
        
        Args:
            query: User query
            context_docs: List of retrieved documents with scores
        
        Returns:
            Generated answer text
        
        Raises:
            Exception: If generation fails
        """
        try:
            # Build prompt with context
            prompt = self._build_prompt(query, context_docs)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text
        
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def get_model_name(self) -> str:
        """
        Get the name of the Gemini model being used
        
        Returns:
            Model name string
        """
        return GEMINI_MODEL