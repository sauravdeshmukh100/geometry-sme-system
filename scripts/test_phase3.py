#!/usr/bin/env python
"""
Improved Phase 3 Test Suite - Merged Implementation
Tests all capabilities with better error handling and reporting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
import time
from typing import Dict, Any

from src.llm.rag_llm_pipeline import GeometryTutorPipeline
from src.config import settings

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


def print_separator(title: str, char: str = "="):
    """Print formatted separator."""
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")


def print_result(result: Dict[str, Any], show_metadata: bool = True):
    """Pretty print test result."""
    if result.get('success'):
        print("✓ SUCCESS")
    else:
        print("✗ FAILED")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    if show_metadata and 'metadata' in result:
        print(f"\nMetadata:")
        for key, value in result.get('metadata', {}).items():
            print(f"  {key}: {value}")


def test_system_initialization():
    """Test 0: System initialization and health check."""
    print_separator("TEST 0: System Initialization", "=")
    
    try:
        print("Initializing Geometry Tutor Pipeline...")
        pipeline = GeometryTutorPipeline()
        
        print("\nGetting system statistics...")
        stats = pipeline.get_statistics()
        
        print("\n" + "─"*70)
        print("SYSTEM STATUS:")
        print(json.dumps(stats, indent=2))
        print("─"*70)
        
        return pipeline
        
    except Exception as e:
        print(f"\n✗ INITIALIZATION FAILED: {e}")
        logger.error(f"System initialization failed: {e}", exc_info=True)
        return None


def test_question_answering(pipeline: GeometryTutorPipeline):
    """Test 1: Question answering with RAG context."""
    print_separator("TEST 1: Question Answering", "=")
    
    test_cases = [
        {
            "name": "Basic Theorem",
            "query": "What is the Pythagorean theorem?",
            "grade": "Grade 9",
            "difficulty": None
        },
        {
            "name": "Triangle Properties",
            "query": "Explain the properties of isosceles triangles",
            "grade": "Grade 7",
            "difficulty": "Beginner"
        },
        {
            "name": "Circle Area",
            "query": "How do I calculate the area of a circle?",
            "grade": "Grade 8",
            "difficulty": "Intermediate"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Grade: {test['grade']}")
        if test['difficulty']:
            print(f"Difficulty: {test['difficulty']}")
        print("─"*70 + "\n")
        
        try:
            start = time.time()
            result = pipeline.answer_question(
                query=test['query'],
                grade_level=test['grade'],
                difficulty=test['difficulty'],
                top_k=5,
                include_sources=True
            )
            elapsed = time.time() - start
            
            if result.get('success'):
                print("ANSWER:")
                print(result['answer'][:400] + "..." if len(result['answer']) > 400 else result['answer'])
                
                if 'sources' in result:
                    print(f"\nSOURCES ({len(result['sources'])}):")
                    for src in result['sources']:
                        print(f"  - {src.get('source')} ({src.get('grade_level')}) [Score: {src.get('score', 0):.4f}]")
                
                print(f"\nRETRIEVAL:")
                retrieval_meta = result.get('retrieval_metadata', {})
                print(f"  Chunks used: {retrieval_meta.get('num_chunks', 0)}")
                print(f"  Avg score: {retrieval_meta.get('avg_score', 0):.4f}")
                print(f"  Strategy: {retrieval_meta.get('strategy', 'unknown')}")
                
                print(f"\nTIMING:")
                print(f"  Total: {elapsed:.2f}s")
                print(f"  Generation: {result.get('generation_metadata', {}).get('generation_time', 0):.2f}s")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown error')}")
            
            results.append({
                'test': test['name'],
                'success': result.get('success'),
                'time': elapsed
            })
            
        except Exception as e:
            print(f"✗ Exception: {e}")
            logger.error(f"Q&A test failed: {e}", exc_info=True)
            results.append({
                'test': test['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    passed = sum(1 for r in results if r.get('success'))
    print(f"Passed: {passed}/{len(results)}")
    print(f"Average time: {sum(r.get('time', 0) for r in results) / len(results):.2f}s")
    print("="*70)


def test_quiz_generation(pipeline: GeometryTutorPipeline):
    """Test 2: Quiz generation with structured output."""
    print_separator("TEST 2: Quiz Generation", "=")
    
    test_cases = [
        {
            "name": "Basic Quiz - Triangles",
            "topic": "Triangles",
            "grade": "Grade 8",
            "num_questions": 3,
            "structured": True
        },
        {
            "name": "Advanced Quiz - Circles",
            "topic": "Circles and their properties",
            "grade": "Grade 9",
            "num_questions": 5,
            "structured": False
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}: {test['name']}")
        print(f"Topic: {test['topic']}")
        print(f"Grade: {test['grade']}")
        print(f"Questions: {test['num_questions']}")
        print(f"Structured output: {test['structured']}")
        print("─"*70 + "\n")
        
        try:
            result = pipeline.generate_quiz(
                topic=test['topic'],
                grade_level=test['grade'],
                num_questions=test['num_questions'],
                use_structured_output=test['structured']
            )
            
            if result.get('success'):
                print("✓ Quiz generated successfully!\n")
                
                if test['structured']:
                    # Display structured quiz
                    print(f"Title: {result.get('quiz_title', 'Geometry Quiz')}")
                    print(f"Topic: {result.get('topic', test['topic'])}\n")
                    
                    for q in result.get('questions', [])[:3]:  # Show first 3
                        print(f"Q{q.get('question_number')}: {q.get('question_text')}")
                        
                        if q.get('question_type') == 'Multiple Choice':
                            for opt in q.get('options', []):
                                print(f"   {opt}")
                        
                        print(f"   ✓ Answer: {q.get('correct_answer')}")
                        print(f"   Explanation: {q.get('explanation', '')[:100]}...\n")
                else:
                    # Display freeform quiz
                    quiz_text = result.get('quiz', '')
                    print(quiz_text[:600] + "..." if len(quiz_text) > 600 else quiz_text)
                
                # Metadata
                metadata = result.get('metadata', {})
                print(f"\nSources: {', '.join(metadata.get('sources', [])[:3])}")
                print(f"Generation time: {metadata.get('generation_time', 0):.2f}s")
            else:
                print(f"✗ Error: {result.get('error')}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            logger.error(f"Quiz generation failed: {e}", exc_info=True)


def test_concept_explanation(pipeline: GeometryTutorPipeline):
    """Test 3: Detailed concept explanations."""
    print_separator("TEST 3: Concept Explanation", "=")
    
    test_cases = [
        {
            "name": "Step-by-step - Pythagorean Theorem",
            "concept": "Pythagorean Theorem",
            "grade": "Grade 9",
            "type": "step-by-step"
        },
        {
            "name": "Visual - Properties of Circles",
            "concept": "Properties of circles",
            "grade": "Grade 8",
            "type": "visual"
        },
        {
            "name": "Example-based - Triangle Similarity",
            "concept": "Triangle similarity",
            "grade": "Grade 10",
            "type": "example"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}: {test['name']}")
        print(f"Concept: {test['concept']}")
        print(f"Grade: {test['grade']}")
        print(f"Type: {test['type']}")
        print("─"*70 + "\n")
        
        try:
            result = pipeline.explain_concept(
                concept=test['concept'],
                grade_level=test['grade'],
                explanation_type=test['type'],
                include_examples=True
            )
            
            if result.get('success'):
                explanation = result.get('explanation', '')
                print("EXPLANATION:")
                print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
                
                metadata = result.get('metadata', {})
                print(f"\nSources: {', '.join(metadata.get('sources', [])[:2])}")
                print(f"Topics: {', '.join(metadata.get('topics', [])[:3])}")
            else:
                print(f"✗ Error: {result.get('error')}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            logger.error(f"Explanation failed: {e}", exc_info=True)


def test_chat_conversation(pipeline: GeometryTutorPipeline):
    """Test 4: Chat with conversation memory."""
    print_separator("TEST 4: Chat Conversation", "=")
    
    messages = [
        "Hi, can you help me with geometry?",
        "What are the different types of triangles?",
        "Can you explain scalene triangles in more detail?",
        "Give me an example problem about scalene triangles"
    ]
    
    chat_history = []
    
    print("Starting conversation...\n")
    
    for i, msg in enumerate(messages, 1):
        print(f"{'─'*70}")
        print(f"Turn {i}")
        print(f"{'─'*70}")
        print(f"Student: {msg}\n")
        
        try:
            result = pipeline.chat(
                message=msg,
                chat_history=chat_history,
                retrieve_context=True,
                grade_level="Grade 8"
            )
            
            if result.get('success'):
                response = result.get('response', '')
                print(f"Tutor: {response[:300]}...")
                
                # Show metadata for first message
                if i == 1:
                    metadata = result.get('metadata', {})
                    print(f"\nMetadata:")
                    print(f"  Retrieval used: {metadata.get('retrieval_used')}")
                    if metadata.get('retrieval_metadata'):
                        ret_meta = metadata['retrieval_metadata']
                        print(f"  Chunks: {ret_meta.get('num_chunks', 0)}")
                        print(f"  Sources: {ret_meta.get('sources', [])}")
                
                chat_history = result.get('chat_history', [])
                print(f"\nHistory length: {len(chat_history)}\n")
            else:
                print(f"✗ Error: {result.get('error')}\n")
                
        except Exception as e:
            print(f"✗ Exception: {e}\n")
            logger.error(f"Chat failed: {e}", exc_info=True)


def test_batch_processing(pipeline: GeometryTutorPipeline):
    """Test 5: Batch question processing."""
    print_separator("TEST 5: Batch Processing", "=")
    
    questions = [
        {"query": "What is a right angle?", "grade_level": "Grade 6"},
        {"query": "Explain parallel lines", "grade_level": "Grade 7"},
        {"query": "How to find area of a trapezoid?", "grade_level": "Grade 8"}
    ]
    
    print(f"Processing {len(questions)} questions in batch...\n")
    
    try:
        start = time.time()
        results = pipeline.batch_answer_questions(
            questions=questions,
            show_progress=True
        )
        elapsed = time.time() - start
        
        print(f"\n{'─'*70}")
        print("BATCH RESULTS:")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average per question: {elapsed/len(questions):.2f}s")
        
        successful = sum(1 for r in results if r.get('success'))
        print(f"Success rate: {successful}/{len(questions)} ({100*successful/len(questions):.1f}%)")
        print("─"*70)
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        logger.error(f"Batch processing failed: {e}", exc_info=True)


def run_interactive_mode(pipeline: GeometryTutorPipeline):
    """Interactive Q&A session."""
    print_separator("INTERACTIVE MODE", "=")
    
    print("Geometry Tutor - Interactive Session")
    print("Ask any geometry question (or type 'quit' to exit)\n")
    
    chat_history = []
    
    while True:
        try:
            query = input("\n🎓 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Keep learning geometry! 📐\n")
                break
            
            if not query:
                continue
            
            # Optional: ask for grade
            grade = input("Grade level (6-10, or Enter for auto): ").strip()
            grade_level = f"Grade {grade}" if grade.isdigit() and 6 <= int(grade) <= 10 else None
            
            print("\n🤔 Thinking...\n")
            
            result = pipeline.answer_question(
                query=query,
                grade_level=grade_level,
                top_k=5
            )
            
            if result.get('success'):
                print("📚 ANSWER:")
                print(result['answer'])
                
                if result.get('sources'):
                    print(f"\n📖 Sources: {', '.join([s['source'] for s in result['sources'][:2]])}")
            else:
                print(f"❌ Error: {result.get('error')}")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye! 📐\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Run comprehensive test suite."""
    print("="*70)
    print("  PHASE 3: LLM INTEGRATION - COMPREHENSIVE TEST SUITE")
    print("  Merged Implementation with Enhanced Features")
    print("="*70)
    
    # Initialize system
    pipeline = test_system_initialization()
    
    if not pipeline:
        print("\n❌ Cannot continue without initialized pipeline")
        return 1
    
    print("\n" + "="*70)
    print("  Select test to run:")
    print("="*70)
    print("1. Question Answering")
    print("2. Quiz Generation")
    print("3. Concept Explanation")
    print("4. Chat Conversation")
    print("5. Batch Processing")
    print("6. All Tests (1-5)")
    print("7. Interactive Mode")
    print("0. Exit")
    
    choice = input("\nYour choice (0-7): ").strip()
    
    try:
        if choice == '1':
            test_question_answering(pipeline)
        elif choice == '2':
            test_quiz_generation(pipeline)
        elif choice == '3':
            test_concept_explanation(pipeline)
        elif choice == '4':
            test_chat_conversation(pipeline)
        elif choice == '5':
            test_batch_processing(pipeline)
        elif choice == '6':
            test_question_answering(pipeline)
            test_quiz_generation(pipeline)
            test_concept_explanation(pipeline)
            test_chat_conversation(pipeline)
            test_batch_processing(pipeline)
        elif choice == '7':
            run_interactive_mode(pipeline)
        elif choice == '0':
            print("\nGoodbye!")
            return 0
        else:
            print("\n❌ Invalid choice")
            return 1
        
        print("\n" + "="*70)
        print("  ✓ TEST SUITE COMPLETED")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        logger.error(f"Test suite failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())