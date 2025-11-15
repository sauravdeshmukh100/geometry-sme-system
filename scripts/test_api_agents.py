#!/usr/bin/env python
"""
API and Agent Integration Tests
Tests FastAPI endpoints and agent functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"


def print_separator(title: str, char: str = "="):
    """Print formatted separator."""
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")


def test_health_check():
    """Test health check endpoint."""
    print_separator("TEST 1: Health Check", "=")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check passed")
            print(f"\nSystem Status: {data.get('status')}")
            print(f"RAG Status: {data.get('system', {}).get('rag_status')}")
            print(f"Total Chunks: {data.get('system', {}).get('total_chunks')}")
            print(f"\nTools Available:")
            for tool, status in data.get('system', {}).get('tools_available', {}).items():
                print(f"  {tool}: {'✓' if status else '✗'}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server")
        print("Make sure the server is running: python src/api/main.py")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_capabilities():
    """Test capabilities endpoint."""
    print_separator("TEST 2: System Capabilities", "=")
    
    try:
        response = requests.get(f"{BASE_URL}/capabilities")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Capabilities retrieved\n")
            
            print("Available Tasks:")
            for task in data.get('tasks', []):
                print(f"\n• {task.get('description')}")
                print(f"  Type: {task.get('type')}")
                print(f"  Example: \"{task.get('example')}\"")
            
            print("\n" + "─"*70)
            print("Features:")
            for feature, enabled in data.get('features', {}).items():
                status = "✓" if enabled else "✗"
                print(f"  {status} {feature}")
            
            print("\n" + "─"*70)
            print(f"Supported Grades: {', '.join(data.get('supported_grades', []))}")
        else:
            print(f"✗ Failed: {response.status_code}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def test_chat_conversation():
    """Test chat endpoint with multi-turn conversation."""
    print_separator("TEST 3: Chat Conversation", "=")
    
    messages = [
        "Hi! Can you help me with geometry?",
        "What is the Pythagorean theorem?",
        "Can you explain it step-by-step?",
        "Give me an example problem"
    ]
    
    try:
        for i, msg in enumerate(messages, 1):
            print(f"\n{'─'*70}")
            print(f"Turn {i}")
            print(f"{'─'*70}")
            print(f"User: {msg}")
            
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"message": msg, "include_context": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\nAssistant ({data.get('type')}):")
                message = data.get('message', '')
                print(message[:300] + "..." if len(message) > 300 else message)
                
                # Show routing info for first message
                if i == 1:
                    routing = data.get('routing', {})
                    print(f"\nRouting Info:")
                    print(f"  Task Type: {routing.get('task_type')}")
                    print(f"  Confidence: {routing.get('confidence', 0):.2f}")
                    print(f"  Workflow: {routing.get('suggested_workflow')}")
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
        
        print("\n✓ Conversation test complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error("Chat test failed", exc_info=True)


def test_quiz_generation():
    """Test quiz generation endpoint."""
    print_separator("TEST 4: Quiz Generation", "=")
    
    quiz_request = {
        "topic": "Triangles",
        "grade_level": "Grade 8",
        "num_questions": 1,
        "format": "pdf"
    }
    
    print(f"Requesting quiz: {json.dumps(quiz_request, indent=2)}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/quiz/generate",
            json=quiz_request
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✓ Quiz generated successfully\n")
            print(f"Topic: {data.get('topic')}")
            print(f"Grade: {data.get('grade_level')}")
            print(f"Questions: {data.get('num_questions')}")
            
            if 'document_path' in data:
                print(f"\n✓ Document created: {data.get('document_path')}")
            
            # Show quiz preview
            if 'quiz' in data:
                quiz_text = data.get('quiz', '')
                print(f"\nQuiz Preview:")
                print(quiz_text[:400] + "..." if len(quiz_text) > 400 else quiz_text)
            
            # Show questions if structured
            if 'questions' in data:
                print(f"\nQuestions ({len(data['questions'])}):")
                for q in data['questions'][:2]:  # Show first 2
                    print(f"\nQ{q.get('question_number')}: {q.get('question_text')}")
                    if q.get('options'):
                        for opt in q.get('options', []):
                            print(f"  {opt}")
        else:
            print(f"✗ Quiz generation failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error("Quiz test failed", exc_info=True)


def test_explanation():
    """Test concept explanation endpoint."""
    print_separator("TEST 5: Concept Explanation", "=")
    
    explain_request = {
        "concept": "Pythagorean Theorem",
        "grade_level": "Grade 9",
        "explanation_type": "step-by-step",
        "include_examples": True
    }
    
    print(f"Requesting explanation: {json.dumps(explain_request, indent=2)}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/explain",
            json=explain_request
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✓ Explanation generated\n")
            print(f"Concept: {data.get('concept')}")
            print(f"Grade: {data.get('grade_level')}")
            print(f"Type: {data.get('type')}")
            
            explanation = data.get('explanation', '')
            print(f"\nExplanation:")
            print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
            
            # Show metadata
            metadata = data.get('metadata', {})
            if metadata.get('sources'):
                print(f"\nSources: {', '.join(metadata['sources'][:2])}")
        else:
            print(f"✗ Explanation failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error("Explanation test failed", exc_info=True)


def test_workflows():
    """Test workflow listing and execution."""
    print_separator("TEST 6: Workflows", "=")
    
    try:
        # List workflows
        response = requests.get(f"{BASE_URL}/workflows")
        
        if response.status_code == 200:
            workflows = response.json()
            
            print("✓ Available Workflows:\n")
            for i, workflow in enumerate(workflows, 1):
                print(f"{i}. {workflow.get('name')}")
                print(f"   Description: {workflow.get('description')}")
                print(f"   Steps: {' → '.join(workflow.get('steps', []))}")
                print(f"   Required: {', '.join(workflow.get('required_params', []))}\n")
        else:
            print(f"✗ Failed to list workflows: {response.status_code}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def test_document_generation():
    """Test document generation endpoint."""
    print_separator("TEST 7: Document Generation", "=")
    
    doc_request = {
        "content_type": "report",
        "format": "pdf",
        "title": "Geometry Study Guide",
        "content": "# Triangles\n\nA triangle is a polygon with three sides...",
        "metadata": {"grade_level": "Grade 8"}
    }
    
    print(f"Requesting document: {doc_request['format']} - {doc_request['title']}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/document/generate",
            json=doc_request
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✓ Document generated\n")
            print(f"Path: {data.get('document_path')}")
            print(f"Format: {data.get('format')}")
            print(f"Download URL: {data.get('download_url')}")
        else:
            print(f"✗ Document generation failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"✗ Error: {e}")


def test_complete_workflow():
    """Test complete quiz generation + email workflow."""
    print_separator("TEST 8: Complete Workflow (Interactive)", "=")
    
    print("This test will generate a quiz and email it.")
    print("Note: Requires email configuration in .env\n")
    
    proceed = input("Proceed? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("Skipping workflow test")
        return
    
    email = input("Enter recipient email: ").strip()
    if not email or '@' not in email:
        print("Invalid email, skipping test")
        return
    
    workflow_request = {
        "topic": "Triangles",
        "grade_level": "Grade 8",
        "num_questions": 5,
        "format": "pdf",
        "to_email": email,
        "student_name": "Test Student",
        "include_html": True
    }
    
    print(f"\nExecuting workflow...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/quiz/email",
            json=workflow_request
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✓ Workflow executed\n")
            print(f"Status: {data.get('overall_status')}")
            print(f"Total time: {data.get('total_time', 0):.2f}s")
            print(f"Steps: {data.get('total_steps')}")
            print(f"Success: {data.get('success_count')}/{data.get('total_steps')}")
            
            # Show step details
            print("\nStep Details:")
            for step in data.get('steps', []):
                status = "✓" if step['status'] == 'success' else "✗"
                print(f"  {status} {step['tool_name']} ({step['execution_time']:.2f}s)")
                if step.get('error'):
                    print(f"      Error: {step['error']}")
        else:
            print(f"✗ Workflow failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error("Workflow test failed", exc_info=True)


def main():
    """Run all API tests."""
    print("="*70)
    print("  API & AGENT INTEGRATION TEST SUITE")
    print("  Phase 5: FastAPI Server + Agents")
    print("="*70)
    
    print("\n⚠️  Make sure the API server is running:")
    print("   python src/api/main.py\n")
    
    print("Available Tests:")
    print("1. Health Check")
    print("2. System Capabilities")
    print("3. Chat Conversation")
    print("4. Quiz Generation")
    print("5. Concept Explanation")
    print("6. Workflow Listing")
    print("7. Document Generation")
    print("8. Complete Workflow (Interactive)")
    print("9. All Tests")
    print("0. Exit")
    
    choice = input("\nSelect test (0-9): ").strip()
    
    try:
        if choice == '1':
            test_health_check()
        elif choice == '2':
            test_capabilities()
        elif choice == '3':
            test_chat_conversation()
        elif choice == '4':
            test_quiz_generation()
        elif choice == '5':
            test_explanation()
        elif choice == '6':
            test_workflows()
        elif choice == '7':
            test_document_generation()
        elif choice == '8':
            test_complete_workflow()
        elif choice == '9':
            test_health_check()
            test_capabilities()
            test_chat_conversation()
            test_quiz_generation()
            test_explanation()
            test_workflows()
            test_document_generation()
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
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logger.error("Test suite failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())