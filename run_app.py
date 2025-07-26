#!/usr/bin/env python3
"""
Startup script for Medical Report Explainer
This script sets up the environment and runs the Streamlit app
"""

import os
import sys
import subprocess
from pathlib import Path

def check_api_key():
    """Check if Gemini API key is set"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("\n📋 To set your API key:")
        print("   Windows (PowerShell): $env:GEMINI_API_KEY=\"your_api_key_here\"")
        print("   Windows (CMD): set GEMINI_API_KEY=your_api_key_here")
        print("   Linux/Mac: export GEMINI_API_KEY=\"your_api_key_here\"")
        print("\n🔑 Get your free API key at: https://makersuite.google.com/app/apikey")
        return False
    else:
        print(f"✅ Gemini API key configured: {api_key[:20]}...")
        return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import faiss
        import sentence_transformers
        import google.generativeai
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False

def setup_data():
    """Setup data directories and create sample corpus if needed"""
    from config.config import DATA_DIR, CORPUS_DIR
    from src.data_ingestion import create_medical_corpus
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CORPUS_DIR.mkdir(exist_ok=True)
    
    # Check if medical corpus exists
    corpus_file = CORPUS_DIR / "medical_corpus.json"
    if not corpus_file.exists():
        print("📚 Medical corpus not found. Creating sample corpus...")
        try:
            create_medical_corpus()
            print("✅ Medical corpus created successfully")
        except Exception as e:
            print(f"⚠️ Warning: Failed to create medical corpus: {e}")
            print("   The app will still work with limited functionality")

def run_app():
    """Run the Streamlit app"""
    try:
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", 
               "--server.port", "8501", "--server.address", "localhost"]
        
        print("\n🚀 Starting Medical Report Explainer...")
        print("   App will be available at: http://localhost:8501")
        print("   Press Ctrl+C to stop the app")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Medical Report Explainer stopped")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")

def main():
    """Main function"""
    print("🏥 Medical Report Explainer - Startup Script")
    print("=" * 50)
    
    # Check API key
    if not check_api_key():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup data
    setup_data()
    
    # Run app
    run_app()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 