"""
Test script to verify Medical Report Explainer setup
Run this script to check if all components are working correctly
"""

import sys
import os
import tempfile
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    try:
        from config.config import GEMINI_API_KEY
        print("✅ Configuration imported successfully")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    return True


def test_api_key():
    """Test if Gemini API key is configured"""
    print("\n🔑 Testing API key configuration...")
    
    try:
        from config.config import GEMINI_API_KEY
        if GEMINI_API_KEY:
            print("✅ Gemini API key is configured")
            return True
        else:
            print("⚠️ Gemini API key not set (set GEMINI_API_KEY environment variable)")
            return False
    except Exception as e:
        print(f"❌ Error checking API key: {e}")
        return False


def test_document_processing():
    """Test document processing functionality"""
    print("\n📄 Testing document processing...")
    
    try:
        from src.document_processor import DocumentProcessor, create_sample_medical_text
        
        processor = DocumentProcessor()
        sample_text = create_sample_medical_text()
        
        if sample_text and len(sample_text) > 100:
            print("✅ Sample medical text generation works")
        else:
            print("❌ Sample text generation failed")
            return False
        
        # Test text preprocessing
        processed_text = processor.preprocess_medical_text(sample_text)
        if processed_text:
            print("✅ Text preprocessing works")
        else:
            print("❌ Text preprocessing failed")
            return False
        
        # Test section extraction
        sections = processor.extract_medical_sections(sample_text)
        if sections and any(sections.values()):
            print("✅ Medical section extraction works")
        else:
            print("❌ Section extraction failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False


def test_embeddings():
    """Test embedding system (without building full vector store)"""
    print("\n🔍 Testing embedding system...")
    
    try:
        from src.embeddings import MedicalEmbeddings
        
        embedder = MedicalEmbeddings()
        
        # Test model loading
        model = embedder.load_model()
        if model:
            print("✅ Embedding model loaded successfully")
        else:
            print("❌ Failed to load embedding model")
            return False
        
        # Test encoding
        test_texts = ["This is a test medical text about hypertension."]
        embeddings = embedder.encode_texts(test_texts)
        
        if embeddings is not None and len(embeddings) > 0:
            print(f"✅ Text encoding works (dimension: {embeddings.shape[1]})")
        else:
            print("❌ Text encoding failed")
            return False
        
        # Test chunking
        sample_text = "This is a long medical text. " * 50
        chunks = embedder.create_chunks(sample_text)
        
        if chunks and len(chunks) > 1:
            print(f"✅ Text chunking works ({len(chunks)} chunks)")
        else:
            print("❌ Text chunking failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False


def test_llm_integration():
    """Test LLM integration (without making API calls)"""
    print("\n🧠 Testing LLM integration...")
    
    try:
        from src.llm_integration import GeminiLLM, MedicalExplainer
        
        # Test LLM initialization
        llm = GeminiLLM()
        if llm:
            print("✅ Gemini LLM initialized")
        else:
            print("❌ Failed to initialize Gemini LLM")
            return False
        
        # Test medical explainer initialization
        explainer = MedicalExplainer()
        if explainer:
            print("✅ Medical explainer initialized")
        else:
            print("❌ Failed to initialize medical explainer")
            return False
        
        # Test term identification
        test_text = "Patient has hypertension and diabetes mellitus type 2."
        complex_terms = explainer.identify_complex_terms(test_text)
        
        if complex_terms and len(complex_terms) > 0:
            print(f"✅ Complex term identification works (found {len(complex_terms)} terms)")
        else:
            print("⚠️ No complex terms identified (this may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False


def test_feedback_system():
    """Test feedback system"""
    print("\n👍 Testing feedback system...")
    
    try:
        from utils.feedback import FeedbackManager
        
        # Create temporary feedback file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_feedback_file = Path(temp_dir) / "test_feedback.json"
            
            feedback_manager = FeedbackManager(feedback_file=temp_feedback_file)
            
            # Test adding feedback
            success = feedback_manager.add_feedback(
                feedback_type="test",
                rating="thumbs_up",
                query="test query",
                response="test response"
            )
            
            if success:
                print("✅ Feedback storage works")
            else:
                print("❌ Feedback storage failed")
                return False
            
            # Test getting stats
            stats = feedback_manager.get_feedback_stats()
            if stats and stats.get("total_feedback", 0) > 0:
                print("✅ Feedback statistics work")
            else:
                print("❌ Feedback statistics failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feedback system test failed: {e}")
        return False


def test_data_directories():
    """Test if data directories are created correctly"""
    print("\n📁 Testing data directory structure...")
    
    try:
        from config.config import DATA_DIR, CORPUS_DIR, EMBEDDINGS_DIR, MODELS_DIR
        
        directories = [DATA_DIR, CORPUS_DIR, EMBEDDINGS_DIR, MODELS_DIR]
        
        for directory in directories:
            if directory.exists():
                print(f"✅ {directory.name}/ directory exists")
            else:
                print(f"❌ {directory.name}/ directory missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🏥 Medical Report Explainer - Setup Test\n")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("API Key Test", test_api_key),
        ("Directory Structure", test_data_directories),
        ("Document Processing", test_document_processing),
        ("Embeddings System", test_embeddings),
        ("LLM Integration", test_llm_integration),
        ("Feedback System", test_feedback_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n🚀 You can now run the application with:")
        print("   streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        print("\n📖 Refer to the README.md for troubleshooting help.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 