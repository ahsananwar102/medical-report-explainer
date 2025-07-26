#!/usr/bin/env python3
"""
Simple test script to verify Gemini API integration
"""

import os
from src.llm_integration import GeminiLLM

def test_gemini_api():
    """Test Gemini API connection"""
    print("🧪 Testing Gemini API connection...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    # Test API call
    llm = GeminiLLM()
    
    test_prompt = "Explain what hypertension means in simple terms."
    
    print("📤 Sending test request...")
    response = llm.generate_response(test_prompt, max_tokens=200)
    
    if response.startswith("Error:"):
        print(f"❌ API test failed: {response}")
        return False
    else:
        print("✅ API test successful!")
        print(f"📝 Response: {response[:100]}...")
        return True

if __name__ == "__main__":
    success = test_gemini_api()
    if success:
        print("\n🎉 Gemini API is working correctly!")
    else:
        print("\n💥 Gemini API test failed!") 