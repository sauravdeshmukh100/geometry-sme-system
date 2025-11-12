#!/usr/bin/env python
"""Test Phase 3: LLM Integration"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.llm.rag_llm_pipeline import GeometryTutorPipeline
from src.config.settings import Settings
import logging

# Setup logging
log_file = getattr(settings, 'log_file', os.path.join(os.path.dirname(__file__), '..', 'logs', 'test_phase3.log'))
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_answer_question():
    """Test Q&A functionality."""
    print("\n" + "="*70)
    print("TEST 1: Question Answering")
    print("="*70)
    
    pipeline = GeometryTutorPipeline()
    
    questions = [
        ("What is the Pythagorean theorem?", "Grade 9"),
        ("Explain properties of triangles", "Grade 7"),
        ("How to calculate area of a circle?", "Grade 8")
    ]
    
    for query, grade in questions:
        print(f"\nQuery: {query} ({grade})")
        print("-" * 70)
        
        result = pipeline.answer_question(
            query=query,
            grade_level=grade,
            top_k=3
        )
        
        if result['success']:
            print(f"✓ Answer generated:")
            print(result['answer'][:300] + "...")
            print(f"\nSources: {result['retrieval_metadata']['sources']}")
        else:
            print(f"✗ Error: {result.get('error')}")

def test_quiz_generation():
    """Test quiz generation."""
    print("\n" + "="*70)
    print("TEST 2: Quiz Generation")
    print("="*70)
    
    pipeline = GeometryTutorPipeline()
    
    result = pipeline.generate_quiz(
        topic="Triangles",
        grade_level="Grade 8",
        num_questions=3
    )
    
    if result['success']:
        print("✓ Quiz generated:")
        print(result['quiz'][:500] + "...")
    else:
        print(f"✗ Error: {result.get('error')}")

def test_explanation():
    """Test concept explanation."""
    print("\n" + "="*70)
    print("TEST 3: Concept Explanation")
    print("="*70)
    
    pipeline = GeometryTutorPipeline()
    
    result = pipeline.explain_concept(
        concept="Pythagorean Theorem",
        grade_level="Grade 9",
        explanation_type="step-by-step"
    )
    
    if result['success']:
        print("✓ Explanation generated:")
        print(result['explanation'][:400] + "...")
    else:
        print(f"✗ Error: {result.get('error')}")

def test_chat():
    """Test conversational interaction."""
    print("\n" + "="*70)
    print("TEST 4: Chat Conversation")
    print("="*70)
    
    pipeline = GeometryTutorPipeline()
    
    messages = [
        "Hi, can you help me with geometry?",
        "What are the types of triangles?",
        "Can you explain scalene triangles?"
    ]
    
    chat_history = []
    
    for msg in messages:
        print(f"\nStudent: {msg}")
        
        result = pipeline.chat(
            message=msg,
            chat_history=chat_history,
            retrieve_context=True
        )
        
        if result['success']:
            print(f"Tutor: {result['response'][:200]}...")
            chat_history = result['chat_history']
        else:
            print(f"✗ Error: {result.get('error')}")

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE 3 - LLM INTEGRATION TESTS")
    print("="*70)
    
    try:
        test_answer_question()
        test_quiz_generation()
        test_explanation()
        test_chat()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()