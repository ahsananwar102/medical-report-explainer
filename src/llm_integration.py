"""
LLM integration module using Google Gemini API
Handles medical text analysis, summarization, and Q&A
"""

import os
import requests
import json
from typing import Dict, List, Optional
import streamlit as st
from dataclasses import dataclass

from config.config import GEMINI_API_KEY, GEMINI_MODEL, READING_LEVELS
from src.embeddings import MedicalRetriever


@dataclass
class MedicalQuery:
    """Structure for medical queries"""
    text: str
    reading_level: str
    query_type: str  # 'explanation', 'summary', 'term_meaning'
    context: Optional[str] = None


class GeminiLLM:
    """Google Gemini API integration for medical text processing"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        """Initialize Gemini API client"""
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            st.error("GEMINI_API_KEY environment variable not set")
    
    def _make_request(self, prompt: str, max_tokens: int = 1000) -> Dict:
        """
        Make request to Gemini API
        
        Args:
            prompt: Text prompt for the model
            max_tokens: Maximum tokens in response
            
        Returns:
            API response dictionary
        """
        if not self.api_key:
            return {"error": "API key not configured"}
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key as query parameter (Gemini API format)
        url += f"?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for medical accuracy
                "topP": 0.8,
                "topK": 40
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Better error handling
            if response.status_code != 200:
                error_details = ""
                try:
                    error_json = response.json()
                    error_details = error_json.get("error", {}).get("message", str(error_json))
                except:
                    error_details = response.text
                
                return {"error": f"API request failed: {response.status_code} - {error_details}"}
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate response from Gemini
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response tokens
            
        Returns:
            Generated text response
        """
        response = self._make_request(prompt, max_tokens)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            # Extract text from Gemini response format
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "No response generated")
            
            return "No response generated"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {str(e)}"


class MedicalExplainer:
    """Medical text explainer using RAG and LLM"""
    
    def __init__(self):
        """Initialize medical explainer with LLM and retriever"""
        self.llm = GeminiLLM()
        self.retriever = MedicalRetriever()
        self.retriever.setup_retriever()
    
    def explain_medical_term(self, term: str, reading_level: str = "8th-grade", 
                           context: str = "") -> str:
        """
        Explain a medical term using RAG
        
        Args:
            term: Medical term to explain
            reading_level: Target reading level
            context: Additional context from medical report
            
        Returns:
            Explanation of the medical term
        """
        # Retrieve relevant medical knowledge
        retrieval_results = self.retriever.retrieve_context(term, max_results=3)
        retrieved_context = self.retriever.format_context_for_llm(retrieval_results)
        
        # Create prompt
        reading_instruction = READING_LEVELS.get(reading_level, READING_LEVELS["8th-grade"])
        
        prompt = f"""You are a medical education assistant. Your task is to explain medical terms clearly and accurately.

Medical Term to Explain: {term}

Context from Patient Report:
{context}

Retrieved Medical Knowledge:
{retrieved_context}

Reading Level Instructions: {reading_instruction}

Please provide a clear, accurate explanation of the medical term "{term}". Include:
1. What the term means in simple language
2. Why it's relevant in medical context
3. Any important details patients should understand

Important: This is for educational purposes only and not medical advice."""
        
        response = self.llm.generate_response(prompt, max_tokens=500)
        return response
    
    def generate_summary(self, medical_text: str, reading_level: str = "8th-grade") -> str:
        """
        Generate a summary of medical text
        
        Args:
            medical_text: Medical report text
            reading_level: Target reading level
            
        Returns:
            Summary of the medical text
        """
        # Extract key medical terms for context retrieval
        key_terms = self._extract_key_terms(medical_text)
        
        # Retrieve relevant context
        if key_terms:
            retrieval_query = " ".join(key_terms[:3])  # Use top 3 terms
            retrieval_results = self.retriever.retrieve_context(retrieval_query, max_results=2)
            retrieved_context = self.retriever.format_context_for_llm(retrieval_results)
        else:
            retrieved_context = "No specific medical context retrieved."
        
        reading_instruction = READING_LEVELS.get(reading_level, READING_LEVELS["8th-grade"])
        
        prompt = f"""You are a medical education assistant. Create a clear summary of this medical report.

Medical Report:
{medical_text}

Retrieved Medical Knowledge:
{retrieved_context}

Reading Level Instructions: {reading_instruction}

Please provide a clear summary that includes:
1. What the main medical issues are
2. What tests or examinations were done
3. What the findings mean
4. What the treatment plan is (if mentioned)

Keep the summary accurate but easy to understand. This is for educational purposes only."""
        
        response = self.llm.generate_response(prompt, max_tokens=800)
        return response
    
    def answer_medical_question(self, question: str, medical_text: str = "", 
                              reading_level: str = "8th-grade") -> str:
        """
        Answer questions about medical content
        
        Args:
            question: User's question
            medical_text: Medical report context
            reading_level: Target reading level
            
        Returns:
            Answer to the question
        """
        # Retrieve relevant medical knowledge
        retrieval_results = self.retriever.retrieve_context(question, max_results=3)
        retrieved_context = self.retriever.format_context_for_llm(retrieval_results)
        
        reading_instruction = READING_LEVELS.get(reading_level, READING_LEVELS["8th-grade"])
        
        prompt = f"""You are a medical education assistant. Answer the following question accurately and clearly.

Question: {question}

Medical Report Context:
{medical_text}

Retrieved Medical Knowledge:
{retrieved_context}

Reading Level Instructions: {reading_instruction}

Please provide a clear, accurate answer to the question. Base your response on the medical knowledge and context provided.

Important reminders:
- This is for educational purposes only
- Always recommend consulting healthcare professionals for medical advice
- Be clear about limitations of the information provided"""
        
        response = self.llm.generate_response(prompt, max_tokens=600)
        return response
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key medical terms from text
        
        Args:
            text: Medical text
            
        Returns:
            List of key medical terms
        """
        # Simple keyword extraction (can be enhanced with NER)
        medical_keywords = [
            "hypertension", "diabetes", "coronary", "myocardial", "infarction",
            "pneumonia", "bronchitis", "asthma", "arthritis", "fracture",
            "surgery", "medication", "treatment", "diagnosis", "symptoms",
            "blood pressure", "heart rate", "temperature", "oxygen",
            "chest pain", "shortness of breath", "nausea", "fever"
        ]
        
        text_lower = text.lower()
        found_terms = []
        
        for keyword in medical_keywords:
            if keyword in text_lower:
                found_terms.append(keyword)
        
        return found_terms[:10]  # Return top 10 terms
    
    def identify_complex_terms(self, text: str) -> List[Dict[str, str]]:
        """
        Identify complex medical terms using LLM analysis
        
        Args:
            text: Medical text
            
        Returns:
            List of dictionaries with term and suggested explanation
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # Use LLM to identify complex medical terms
        prompt = f"""You are a medical education assistant. Analyze the following medical text and identify complex medical terms that a layperson (non-medical person) would find difficult to understand.

Medical Text:
{text}

Please identify complex medical terms and provide simple explanations. Focus on:
1. Medical terminology and jargon
2. Technical procedures and tests  
3. Medical abbreviations
4. Anatomical terms
5. Drug names and medical treatments
6. Laboratory values and measurements
7. Medical conditions and diagnoses

Format your response as a JSON list where each item has:
- "term": the complex medical term (exactly as it appears in the text)
- "simple_explanation": a simple explanation a 12-year-old could understand
- "category": the type of term (condition, procedure, test, medication, anatomy, etc.)

Example format:
[
  {{"term": "hypertension", "simple_explanation": "high blood pressure", "category": "condition"}},
  {{"term": "electrocardiogram", "simple_explanation": "heart test that shows electrical activity", "category": "test"}}
]

Only include terms that actually appear in the provided text. Limit to the 15 most important complex terms."""

        try:
            # Get LLM response
            response = self.llm.generate_response(prompt, max_tokens=800)
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    llm_terms = json.loads(json_str)
                    
                    # Convert to our expected format and validate
                    found_terms = []
                    text_lower = text.lower()
                    
                    for i, term_data in enumerate(llm_terms):
                        if isinstance(term_data, dict) and 'term' in term_data:
                            term = term_data['term'].lower().strip()
                            
                            # Verify the term actually exists in the text
                            if term in text_lower:
                                position = text_lower.find(term)
                                found_terms.append({
                                    "term": term_data['term'].strip(),  # Keep original casing
                                    "simple_explanation": term_data.get('simple_explanation', 'Medical term requiring explanation'),
                                    "position": position,
                                    "type": "llm_extracted",
                                    "category": term_data.get('category', 'medical_term')
                                })
                    
                    # Sort by position in text
                    found_terms.sort(key=lambda x: x["position"])
                    
                    # If we got good results from LLM, return them
                    if found_terms:
                        return found_terms[:15]  # Limit to top 15
                        
                except json.JSONDecodeError:
                    print("Failed to parse LLM JSON response, falling back to pattern matching")
            
        except Exception as e:
            print(f"Error using LLM for term extraction: {e}")
        
        # Fallback: Use pattern matching for common medical patterns
        return self._fallback_pattern_extraction(text)
    
    def _fallback_pattern_extraction(self, text: str) -> List[Dict[str, str]]:
        """
        Fallback method using pattern matching for medical terms
        
        Args:
            text: Medical text
            
        Returns:
            List of medical terms found using patterns
        """
        import re
        
        # Common medical patterns and abbreviations
        medical_patterns = {
            # Medical suffixes
            r'\b\w+itis\b': "inflammation condition",
            r'\b\w+osis\b': "medical condition", 
            r'\b\w+emia\b': "blood condition",
            r'\b\w+uria\b': "urine condition",
            r'\b\w+pathy\b': "disease condition",
            r'\b\w+cardia\b': "heart condition",
            r'\b\w+pnea\b': "breathing condition",
            r'\b\w+megaly\b': "organ enlargement",
            r'\b\w+scopy\b': "medical examination",
            r'\b\w+ectomy\b': "surgical removal",
            r'\b\w+plasty\b': "surgical repair",
            r'\b\w+graphy\b': "medical imaging",
            r'\b\w+gram\b': "medical record/image",
        }
        
        # Common medical abbreviations
        abbreviations = {
            "ecg": "heart electrical test",
            "ekg": "heart electrical test",
            "mri": "body scan using magnets",
            "ct": "detailed x-ray scan", 
            "cbc": "complete blood count test",
            "bmi": "body weight measurement",
            "bp": "blood pressure",
            "hr": "heart rate",
            "iv": "medicine through vein",
            "icu": "intensive care unit",
            "er": "emergency room",
            "or": "operating room"
        }
        
        # Key medical terms
        key_terms = {
            "hypertension": "high blood pressure",
            "diabetes": "high blood sugar condition", 
            "pneumonia": "lung infection",
            "myocardial infarction": "heart attack",
            "dyspnea": "trouble breathing",
            "tachycardia": "fast heart rate",
            "bradycardia": "slow heart rate",
            "edema": "swelling from fluid",
            "diaphoresis": "excessive sweating",
            "substernal": "under the breastbone",
            "bilateral": "on both sides"
        }
        
        found_terms = []
        text_lower = text.lower()
        
        # Check for exact key terms
        for term, explanation in key_terms.items():
            if term in text_lower:
                position = text_lower.find(term)
                found_terms.append({
                    "term": term,
                    "simple_explanation": explanation,
                    "position": position,
                    "type": "exact_match",
                    "category": "medical_term"
                })
        
        # Check for abbreviations
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        for word in words:
            if word in abbreviations:
                position = text_lower.find(word)
                found_terms.append({
                    "term": word.upper(),
                    "simple_explanation": abbreviations[word],
                    "position": position,
                    "type": "abbreviation",
                    "category": "abbreviation"
                })
        
        # Check for pattern matches
        for pattern, explanation in medical_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if (match not in [t["term"].lower() for t in found_terms] and 
                    len(match) > 3):
                    position = text_lower.find(match)
                    found_terms.append({
                        "term": match,
                        "simple_explanation": explanation,
                        "position": position,
                        "type": "pattern_match",
                        "category": "medical_term"
                    })
        
        # Remove duplicates and sort by position
        seen_terms = set()
        unique_terms = []
        for term in found_terms:
            term_lower = term["term"].lower()
            if term_lower not in seen_terms:
                seen_terms.add(term_lower)
                unique_terms.append(term)
        
        # Sort by position and limit results
        unique_terms.sort(key=lambda x: x["position"])
        return unique_terms[:10]


def create_medical_disclaimer() -> str:
    """Create medical disclaimer text"""
    return """
⚠️ **Important Medical Disclaimer**

This tool is designed for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. The information provided:

- Is for general educational purposes only
- Should not be considered as medical advice
- May not be applicable to your specific medical situation
- Should not replace consultation with qualified healthcare providers

**Always consult with your doctor or other qualified healthcare provider for:**
- Medical advice regarding your health
- Questions about medical conditions
- Treatment decisions
- Medication changes

**In case of medical emergency, contact emergency services immediately.**

The developers and providers of this tool are not responsible for any medical decisions made based on the information provided.
""" 