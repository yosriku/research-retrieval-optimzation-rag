"""
Retrieval Module
Handles FAISS vector database operations and document retrieval
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from config import FAISS_INDEX_FILENAME, PICKLE_FILENAME

class FAISSRetriever:
    """
    FAISS-based document retriever
    Handles loading index, documents, and performing similarity search
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        print(f"--- DEBUG: Initializing Retriever for {db_path} ---")
        
        # 1. Load Index
        self.index = self._load_faiss_index()
        
        # 2. Load Documents (Chunks) dengan logic LangChain yang benar
        self.documents = self._load_documents_from_langchain_pickle()
        
        print(f"--- DEBUG: Index Size: {self.index.ntotal} vectors")
        print(f"--- DEBUG: Doc Store Size: {len(self.documents)} chunks")
        
        # Validasi sinkronisasi
        if self.index.ntotal != len(self.documents):
            print("⚠️ WARNING: Jumlah vector dan chunk tidak sama! Mungkin ada yang hilang.")
    
    def _load_faiss_index(self) -> faiss.Index:
        index_path = os.path.join(self.db_path, FAISS_INDEX_FILENAME)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        try:
            index = faiss.read_index(index_path)
            return index
        except Exception as e:
            raise Exception(f"Error loading FAISS index: {str(e)}")
    
    def _load_documents_from_langchain_pickle(self) -> List[str]:
        """
        Load document chunks from LangChain-generated pickle file.
        LangChain saves: (docstore, index_to_docstore_id)
        """
        pkl_path = os.path.join(self.db_path, PICKLE_FILENAME)
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Document pickle file not found at: {pkl_path}")
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Cek apakah formatnya Tuple (Docstore, Mapping) khas LangChain
            if isinstance(data, tuple) and len(data) == 2:
                docstore, index_to_docstore_id = data
                
                ordered_chunks = []
                # Loop urut sesuai ID integer FAISS (0, 1, 2, ...)
                # index_to_docstore_id adalah dict {0: 'uuid-1', 1: 'uuid-2', ...}
                total_vectors = len(index_to_docstore_id)
                
                for i in range(total_vectors):
                    doc_id = index_to_docstore_id.get(i)
                    if doc_id:
                        document_obj = docstore.search(doc_id)
                        # Ambil HANYA text content (chunk string)
                        ordered_chunks.append(document_obj.page_content)
                    else:
                        ordered_chunks.append("[MISSING DOC]")
                        
                return ordered_chunks
                
            else:
                # Fallback jika ternyata formatnya list biasa (custom)
                return data
                
        except Exception as e:
            raise Exception(f"Error loading documents: {str(e)}")
    
    def retrieve(
        self, 
        query: str, 
        embedding_model: SentenceTransformer, 
        top_k: int = 3,
        distance_threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant documents for a query
        """
        try:
            print(f"\n--- DEBUG RETRIEVAL ---")
            print(f"Query: {query}")
            
            # Encode query
            query_embedding = embedding_model.encode(
                [query], 
                convert_to_numpy=True,
                normalize_embeddings=True 
            )
            
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                top_k
            )
            
            print(f"Indices found: {indices[0]}")
            print(f"Distances: {distances[0]}")
            
            retrieved_docs = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.documents):
                    
                    # Logic Threshold (Opsional, matikan di config jika perlu)
                    if distance_threshold is None or dist <= distance_threshold:
                        # self.documents[idx] sekarang DIJAMIN string chunk
                        retrieved_docs.append((self.documents[idx], float(dist)))
                    else:
                        print(f"Skipped doc {idx} (Distance {dist:.4f} > Threshold {distance_threshold})")
                        
            print(f"Returned: {len(retrieved_docs)} docs")
            return retrieved_docs
        
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception(f"Error during retrieval: {str(e)}")
            
    # Getter helpers
    def get_index_size(self) -> int:
        return self.index.ntotal
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def get_dimension(self) -> int:
        return self.index.d