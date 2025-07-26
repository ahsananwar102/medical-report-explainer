"""
Configuration settings for Medical Report Explainer
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CORPUS_DIR = DATA_DIR / "corpus"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, EMBEDDINGS_DIR, CORPUS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Keys and Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Free model

# Embedding Settings - Using medical-specific model
EMBEDDING_MODEL = "abhinand/MedEmbed-base-v0.1"  # Medical-specific embedding model
EMBEDDING_DIMENSION = 768  # MedEmbed dimension (based on bge-base-en-v1.5)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# FAISS Settings
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "medical_corpus.faiss"
FAISS_METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

# Medical Corpus Settings
PUBMED_EMAIL = "medreport.explainer@example.com"  # Required for Entrez API
MAX_PUBMED_ARTICLES = 1000
MAYO_CLINIC_BASE_URL = "https://www.mayoclinic.org"

# Streamlit Settings
PAGE_TITLE = "Medical Report Explainer"
PAGE_ICON = "🏥"
LAYOUT = "wide"

# Processing Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = [".pdf", ".txt", ".docx"]

# UI Settings
READING_LEVELS = {
    "12-year-old": "Explain this using very simple words that a 12-year-old would understand. Use short sentences and avoid medical jargon. Do not output any other text than the explanation.",
    "8th-grade": "Explain this at an 8th-grade reading level. Use clear, straightforward language with some medical terms explained in parentheses. Do not output any other text than the explanation."
}

# Disclaimer text
DISCLAIMER = """
⚠️ **Medical Disclaimer**: This tool is for educational purposes only and is not intended to provide medical advice, 
diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns. 
The information provided should not replace professional medical consultation.
"""

# Feedback storage
FEEDBACK_FILE = DATA_DIR / "feedback.json" 