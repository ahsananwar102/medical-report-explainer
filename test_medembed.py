#!/usr/bin/env python3
"""
Test script to verify MedEmbed model functionality
"""

import os
import numpy as np
from src.embeddings import MedicalEmbeddings

def test_medembed_model():
    """Test MedEmbed model loading and encoding"""
    print("🧪 Testing MedEmbed medical embedding model...")
    
    try:
        # Initialize embeddings with MedEmbed
        embedder = MedicalEmbeddings()
        
        print("📥 Loading MedEmbed model...")
        model = embedder.load_model()
        
        if model is None:
            print("❌ Failed to load MedEmbed model")
            return False
        
        print(f"✅ MedEmbed model loaded successfully: {embedder.model_name}")
        print(f"📏 Expected dimension: {embedder.dimension}")
        
        # Test with medical texts
        medical_texts = [
            "Patient presents with acute myocardial infarction and elevated troponin levels.",
            "Hypertension is a chronic condition characterized by persistently elevated blood pressure.",
            "The patient shows signs of diabetes mellitus type 2 with poor glycemic control.",
            "Physical examination reveals bilateral crackles and peripheral edema.",
            "Coronary angiography demonstrated significant stenosis in the left anterior descending artery."
        ]
        
        print("🔍 Testing medical text encoding...")
        embeddings = embedder.encode_texts(medical_texts)
        
        if embeddings is None or len(embeddings) == 0:
            print("❌ Failed to encode medical texts")
            return False
        
        print(f"✅ Successfully encoded {len(medical_texts)} medical texts")
        print(f"📐 Embedding shape: {embeddings.shape}")
        print(f"📊 Embedding dimensions: {embeddings.shape[1]}")
        
        # Verify dimensions
        if embeddings.shape[1] != embedder.dimension:
            print(f"⚠️ Warning: Expected dimension {embedder.dimension}, got {embeddings.shape[1]}")
        else:
            print("✅ Embedding dimensions match expected size")
        
        # Test similarity between medical texts
        print("\n🔗 Testing medical concept similarity...")
        
        # Test specific medical concepts
        concept_texts = [
            "myocardial infarction",
            "heart attack", 
            "hypertension",
            "high blood pressure",
            "diabetes mellitus",
            "asthma"
        ]
        
        concept_embeddings = embedder.encode_texts(concept_texts)
        
        # Calculate similarity between related concepts
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Test similarity between "myocardial infarction" and "heart attack"
        mi_embedding = concept_embeddings[0]
        ha_embedding = concept_embeddings[1]
        mi_ha_similarity = cosine_similarity(mi_embedding, ha_embedding)
        
        # Test similarity between "hypertension" and "high blood pressure"
        ht_embedding = concept_embeddings[2]
        bp_embedding = concept_embeddings[3]
        ht_bp_similarity = cosine_similarity(ht_embedding, bp_embedding)
        
        print(f"📈 Myocardial infarction ↔ Heart attack similarity: {mi_ha_similarity:.3f}")
        print(f"📈 Hypertension ↔ High blood pressure similarity: {ht_bp_similarity:.3f}")
        
        # These should be highly similar (> 0.7) for a good medical model
        if mi_ha_similarity > 0.7 and ht_bp_similarity > 0.7:
            print("✅ MedEmbed shows good understanding of medical synonyms")
        else:
            print("⚠️ MedEmbed similarity scores lower than expected")
        
        return True
        
    except Exception as e:
        print(f"❌ MedEmbed test failed: {e}")
        return False

def test_vector_store_rebuild():
    """Test rebuilding vector store with MedEmbed"""
    print("\n🏗️ Testing vector store rebuild with MedEmbed...")
    
    try:
        from src.embeddings import initialize_embeddings_system
        
        # Force rebuild of vector store
        success, message = initialize_embeddings_system()
        
        if success:
            print("✅ Vector store successfully rebuilt with MedEmbed")
        else:
            print(f"❌ Vector store rebuild failed: {message}")
        
        return success
        
    except Exception as e:
        print(f"❌ Vector store rebuild test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 MedEmbed Model Test")
    print("=" * 50)
    
    # Test model functionality
    model_test = test_medembed_model()
    
    if model_test:
        # Test vector store rebuild
        vector_test = test_vector_store_rebuild()
        
        if vector_test:
            print("\n🎉 All MedEmbed tests passed!")
            print("\n💡 Benefits of MedEmbed:")
            print("   • Better understanding of medical terminology")
            print("   • Improved retrieval of relevant medical information")
            print("   • Enhanced performance on medical Q&A tasks")
            print("   • Specialized for clinical and medical contexts")
            return True
        else:
            print("\n⚠️ Model works but vector store rebuild failed")
            return False
    else:
        print("\n❌ MedEmbed model test failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 