"""
Embeddings and vector store module for medical corpus retrieval
Uses sentence-transformers for embeddings and FAISS for similarity search
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Handle imports that might not be available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from config.config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH, CORPUS_DIR
)


def safe_spinner(message: str):
    """Safe spinner that works with or without Streamlit"""
    if HAS_STREAMLIT:
        return st.spinner(message)
    else:
        print(f"⏳ {message}")
        return DummySpinner()

def safe_error(message: str):
    """Safe error display that works with or without Streamlit"""
    if HAS_STREAMLIT:
        st.error(message)
    else:
        print(f"❌ ERROR: {message}")

def safe_success(message: str):
    """Safe success display that works with or without Streamlit"""
    if HAS_STREAMLIT:
        st.success(message)
    else:
        print(f"✅ {message}")

class DummySpinner:
    """Dummy spinner for non-Streamlit environments"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class MedicalEmbeddings:
    """Handle medical text embeddings and similarity search"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize with sentence transformer model"""
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = None
        self.dimension = EMBEDDING_DIMENSION
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            with safe_spinner(f"Loading embedding model: {self.model_name}"):
                try:
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    error_msg = f"Failed to load embedding model: {e}"
                    safe_error(error_msg)
                    raise RuntimeError(error_msg)
        return self.model
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts into embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        model = self.load_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
        except Exception as e:
            error_msg = f"Failed to encode texts: {e}"
            safe_error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_chunks(self, text: str, chunk_size: int = CHUNK_SIZE, 
                     chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def process_medical_corpus(self, corpus_file: Optional[Path] = None) -> List[Document]:
        """
        Process medical corpus into document chunks
        
        Args:
            corpus_file: Path to corpus JSON file
            
        Returns:
            List of LangChain Document objects
        """
        if corpus_file is None:
            corpus_file = CORPUS_DIR / "medical_corpus.json"
        
        if not corpus_file.exists():
            error_msg = f"Medical corpus not found: {corpus_file}"
            safe_error(error_msg)
            return []
        
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
        except Exception as e:
            error_msg = f"Failed to load corpus file: {e}"
            safe_error(error_msg)
            return []
        
        documents = []
        
        # Process PubMed articles
        for article in corpus_data.get("pubmed_articles", []):
            # Combine title and abstract
            full_text = f"{article.get('title', '')}\n\n{article.get('abstract', '')}"
            
            if full_text.strip():
                chunks = self.create_chunks(full_text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "pubmed",
                            "pmid": article.get("pmid", ""),
                            "title": article.get("title", ""),
                            "keywords": article.get("keywords", []),
                            "mesh_terms": article.get("mesh_terms", []),
                            "chunk_id": i
                        }
                    )
                    documents.append(doc)
        
        # Process Mayo Clinic articles
        for article in corpus_data.get("mayo_articles", []):
            full_text = f"{article.get('title', '')}\n\n{article.get('content', '')}"
            
            if full_text.strip():
                chunks = self.create_chunks(full_text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "mayo_clinic",
                            "title": article.get("title", ""),
                            "url": article.get("url", ""),
                            "chunk_id": i
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def build_vector_store(self, documents: Optional[List[Document]] = None) -> bool:
        """
        Build FAISS vector store from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Success status
        """
        if documents is None:
            documents = self.process_medical_corpus()
        
        if not documents:
            safe_error("No documents found to build vector store")
            return False
        
        with safe_spinner(f"Building vector store with {len(documents)} documents..."):
            try:
                # Extract text content
                texts = [doc.page_content for doc in documents]
                
                # Create embeddings
                embeddings = self.encode_texts(texts)
                
                # Create FAISS index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add embeddings to index
                self.index.add(embeddings.astype('float32'))
                
                # Store metadata
                self.metadata = [doc.metadata for doc in documents]
                
                # Save to disk
                self.save_vector_store()
                
                safe_success(f"Vector store built with {len(documents)} documents")
                return True
            except Exception as e:
                error_msg = f"Failed to build vector store: {e}"
                safe_error(error_msg)
                return False
    
    def save_vector_store(self):
        """Save FAISS index and metadata to disk"""
        if self.index is not None:
            try:
                # Ensure directories exist
                FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
                
                # Save FAISS index
                faiss.write_index(self.index, str(FAISS_INDEX_PATH))
                
                # Save metadata
                with open(FAISS_METADATA_PATH, 'wb') as f:
                    pickle.dump(self.metadata, f)
            except Exception as e:
                error_msg = f"Failed to save vector store: {e}"
                safe_error(error_msg)
                raise
    
    def load_vector_store(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            Success status
        """
        try:
            if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                
                # Load metadata
                with open(FAISS_METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                return True
            else:
                return False
        except Exception as e:
            safe_error(f"Error loading vector store: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using query
        
        Args:
            query: Search query text
            k: Number of top results to return
            
        Returns:
            List of similar documents with metadata and scores
        """
        if self.index is None:
            if not self.load_vector_store():
                safe_error("Vector store not available. Please build it first.")
                return []
        
        try:
            # Encode query
            model = self.load_model()
            query_embedding = model.encode([query], convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    result = {
                        "rank": i + 1,
                        "score": float(score),
                        "metadata": self.metadata[idx],
                        "index": int(idx)
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            safe_error(f"Error during similarity search: {e}")
            return []
    
    def get_document_content(self, doc_index: int) -> str:
        """
        Get original document content by index
        
        Args:
            doc_index: Index of document in vector store
            
        Returns:
            Document content
        """
        # This would require storing the original text content
        # For now, return metadata information
        if self.metadata and 0 <= doc_index < len(self.metadata):
            metadata = self.metadata[doc_index]
            return f"Source: {metadata.get('source', 'Unknown')}\nTitle: {metadata.get('title', 'No title')}"
        return "Content not available"


class MedicalRetriever:
    """Enhanced retriever for medical knowledge"""
    
    def __init__(self):
        self.embeddings = MedicalEmbeddings()
        
    def setup_retriever(self) -> bool:
        """Setup the retriever with vector store"""
        return self.embeddings.load_vector_store()
    
    def retrieve_context(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant medical context for query
        
        Args:
            query: User query
            max_results: Maximum number of results
            
        Returns:
            List of relevant context documents
        """
        return self.embeddings.search_similar(query, k=max_results)
    
    def format_context_for_llm(self, context_results: List[Dict]) -> str:
        """
        Format retrieved context for LLM consumption
        
        Args:
            context_results: Results from retrieval
            
        Returns:
            Formatted context string
        """
        if not context_results:
            return "No relevant medical information found."
        
        formatted_context = "Relevant Medical Information:\n\n"
        
        for i, result in enumerate(context_results):
            metadata = result["metadata"]
            formatted_context += f"{i+1}. Source: {metadata.get('source', 'Unknown')}\n"
            
            if metadata.get('title'):
                formatted_context += f"   Title: {metadata['title']}\n"
            
            if metadata.get('pmid'):
                formatted_context += f"   PMID: {metadata['pmid']}\n"
            
            # Add keywords or MeSH terms if available
            if metadata.get('keywords'):
                formatted_context += f"   Keywords: {', '.join(metadata['keywords'][:5])}\n"
            
            formatted_context += f"   Relevance Score: {result['score']:.3f}\n\n"
        
        return formatted_context


def initialize_embeddings_system() -> Tuple[bool, str]:
    """
    Initialize the embeddings system
    
    Returns:
        Tuple of (success, message)
    """
    try:
        embeddings = MedicalEmbeddings()
        
        # Try to load existing vector store
        if embeddings.load_vector_store():
            return True, "Vector store loaded successfully"
        
        # If no vector store exists, build one
        corpus_file = CORPUS_DIR / "medical_corpus.json"
        if corpus_file.exists():
            success = embeddings.build_vector_store()
            if success:
                return True, "Vector store built successfully"
            else:
                return False, "Failed to build vector store"
        else:
            return False, f"Medical corpus not found at {corpus_file}. Please run data ingestion first."
    except Exception as e:
        return False, f"Error initializing embeddings system: {e}" 