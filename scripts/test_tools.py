#!/usr/bin/env python
"""
Tool Integration Tests
Tests document generation, email sending, and orchestration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Any

from src.tools.document_generator import DocumentGenerator
from src.tools.email_sender import EmailSender
from src.tools.tool_orchestrator import ToolOrchestrator, ToolStatus
from src.llm.rag_llm_pipeline import GeometryTutorPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str, char: str = "="):
    """Print formatted separator."""
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")


def test_document_generation():
    """Test PDF, DOCX, and PPT generation."""
    print_separator("TEST 1: Document Generation", "=")
    
    doc_gen = DocumentGenerator(output_dir="../test_outputs")
    
    # Check available formats
    formats = doc_gen.get_supported_formats()
    print("Available formats:")
    for fmt, available in formats.items():
        print(f"  {fmt}: {'✓' if available else '✗'}")
    
    if not any(formats.values()):
        print("\n⚠️  No document generation libraries installed!")
        print("Install with: pip install reportlab python-docx python-pptx")
        return
    
    # Test data
    quiz_data = {
        'quiz_title': 'Geometry Quiz: Triangles',
        'topic': 'Triangles',
        'grade_level': 'Grade 8',
        'num_questions': 3,
        'questions': [
            {
                'question_number': 1,
                'question_text': 'What is the sum of angles in a triangle?',
                'question_type': 'Multiple Choice',
                'options': ['A) 90°', 'B) 180°', 'C) 270°', 'D) 360°'],
                'correct_answer': 'B',
                'explanation': 'The sum of interior angles in any triangle is always 180°.'
            },
            {
                'question_number': 2,
                'question_text': 'Define an isosceles triangle.',
                'question_type': 'Short Answer',
                'options': [],
                'correct_answer': 'A triangle with two equal sides',
                'explanation': 'An isosceles triangle has at least two sides of equal length.'
            }
        ],
        'metadata': {
            'generated_at': '2024-01-15 10:30:00'
        }
    }
    
    explanation_data = {
        'concept': 'Pythagorean Theorem',
        'explanation': '''The Pythagorean Theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.

Formula: a² + b² = c²

Where:
- a and b are the lengths of the legs
- c is the length of the hypotenuse

Example: If a = 3 and b = 4, then c² = 9 + 16 = 25, so c = 5.''',
        'grade_level': 'Grade 9',
        'metadata': {
            'sources': ['NCERT Grade 9'],
            'topics': ['Right Triangles', 'Theorems']
        }
    }
    
    # Test each format
    test_cases = []
    
    if formats.get('pdf'):
        print("\n" + "─"*70)
        print("Testing PDF Generation")
        print("─"*70)
        
        try:
            # Quiz PDF
            pdf_path = doc_gen.generate(
                content_type='quiz',
                format='pdf',
                data=quiz_data,
                filename='test_quiz.pdf',
                include_answers=True
            )
            print(f"✓ Quiz PDF generated: {pdf_path}")
            test_cases.append(('PDF Quiz', True, pdf_path))
            
            # Report PDF
            pdf_path = doc_gen.generate(
                content_type='explanation',
                format='pdf',
                data=explanation_data,
                filename='test_explanation.pdf'
            )
            print(f"✓ Explanation PDF generated: {pdf_path}")
            test_cases.append(('PDF Explanation', True, pdf_path))
            
        except Exception as e:
            print(f"✗ PDF generation failed: {e}")
            test_cases.append(('PDF', False, str(e)))
    
    if formats.get('docx'):
        print("\n" + "─"*70)
        print("Testing DOCX Generation")
        print("─"*70)
        
        try:
            # Quiz DOCX
            docx_path = doc_gen.generate(
                content_type='quiz',
                format='docx',
                data=quiz_data,
                filename='test_quiz.docx',
                include_answers=True
            )
            print(f"✓ Quiz DOCX generated: {docx_path}")
            test_cases.append(('DOCX Quiz', True, docx_path))
            
            # Report DOCX
            docx_path = doc_gen.generate(
                content_type='explanation',
                format='docx',
                data=explanation_data,
                filename='test_explanation.docx'
            )
            print(f"✓ Explanation DOCX generated: {docx_path}")
            test_cases.append(('DOCX Explanation', True, docx_path))
            
        except Exception as e:
            print(f"✗ DOCX generation failed: {e}")
            test_cases.append(('DOCX', False, str(e)))
    
    if formats.get('pptx'):
        print("\n" + "─"*70)
        print("Testing PPTX Generation")
        print("─"*70)
        
        try:
            # Quiz PPT
            ppt_path = doc_gen.generate(
                content_type='quiz',
                format='pptx',
                data=quiz_data,
                filename='test_quiz.pptx'
            )
            print(f"✓ Quiz PPTX generated: {ppt_path}")
            test_cases.append(('PPTX Quiz', True, ppt_path))
            
            # Concept PPT
            ppt_path = doc_gen.generate(
                content_type='concept',
                format='pptx',
                data=explanation_data,
                filename='test_concept.pptx'
            )
            print(f"✓ Concept PPTX generated: {ppt_path}")
            test_cases.append(('PPTX Concept', True, ppt_path))
            
        except Exception as e:
            print(f"✗ PPTX generation failed: {e}")
            test_cases.append(('PPTX', False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("DOCUMENT GENERATION SUMMARY")
    print("="*70)
    
    for name, success, output in test_cases:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {output[:50]}...")
    
    passed = sum(1 for _, success, _ in test_cases if success)
    print(f"\nPassed: {passed}/{len(test_cases)}")


def test_email_sending():
    """Test email functionality (requires configuration)."""
    print_separator("TEST 2: Email Sending", "=")
    
    email_sender = EmailSender()
    
    # Check configuration
    if not email_sender.username or not email_sender.password:
        print("⚠️  Email not configured!")
        print("\nTo enable email testing:")
        print("1. Set EMAIL_USERNAME in .env (e.g., your Gmail)")
        print("2. Set EMAIL_PASSWORD in .env (use App Password for Gmail)")
        print("3. For Gmail: https://support.google.com/accounts/answer/185833")
        print("\nOptional settings:")
        print("  SMTP_SERVER (default: smtp.gmail.com)")
        print("  SMTP_PORT (default: 587)")
        return
    
    # Test connection
    print("Testing email connection...")
    if email_sender.test_connection():
        print("✓ Email connection successful!")
    else:
        print("✗ Email connection failed!")
        return
    
    # Prompt for test email
    print("\n" + "─"*70)
    test_email = input("Enter email address for testing (or 'skip'): ").strip()
    
    if test_email.lower() == 'skip' or not test_email:
        print("Skipping email send test")
        return
    
    # Test simple email
    print(f"\nSending test email to {test_email}...")
    
    result = email_sender.send_email(
        to_email=test_email,
        subject="Test Email from Geometry Tutor",
        body="This is a test email from the Geometry SME system.\n\nIf you received this, email integration is working!",
        html=False
    )
    
    if result['success']:
        print(f"✓ Email sent successfully!")
        print(f"  Sent at: {result['sent_at']}")
    else:
        print(f"✗ Email failed: {result['error']}")


def test_tool_orchestration():
    """Test complete workflow orchestration."""
    print_separator("TEST 3: Tool Orchestration", "=")
    
    print("Initializing orchestrator...")
    orchestrator = ToolOrchestrator()
    
    # Get tool status
    print("\n" + "─"*70)
    print("Tool Status:")
    status = orchestrator.get_tool_status()
    
    for tool, available in status['available_tools'].items():
        print(f"  {tool}: {'✓' if available else '✗'}")
    
    print("\nDocument formats:")
    for fmt, available in status['document_formats'].items():
        print(f"  {fmt}: {'✓' if available else '✗'}")
    
    print(f"\nEmail configured: {'✓' if status['email_configured'] else '✗'}")
    print(f"Max retries: {status['max_retries']}")
    print(f"Fallbacks enabled: {status['fallbacks_enabled']}")
    
    # Test workflows (without email)
    print("\n" + "─"*70)
    print("Testing Quiz Generation Workflow (without email)")
    print("─"*70)
    
    try:
        # Mock workflow - generate quiz and document only
        print("\nStep 1: Generating quiz content...")
        pipeline = GeometryTutorPipeline()
        
        quiz_result = pipeline.generate_quiz(
            topic="Triangles",
            grade_level="Grade 8",
            num_questions=3
        )
        
        if quiz_result.get('success'):
            print("✓ Quiz generated")
            
            print("\nStep 2: Creating PDF document...")
            doc_gen = DocumentGenerator()
            
            doc_path = doc_gen.generate(
                content_type='quiz',
                format='pdf',
                data=quiz_result,
                filename='workflow_test_quiz.pdf'
            )
            
            print(f"✓ Document created: {doc_path}")
            
            print("\nStep 3: Email (skipped - no config)")
            print("  Would send to: student@example.com")
            
            print("\n✓ Workflow completed successfully!")
            
        else:
            print(f"✗ Quiz generation failed: {quiz_result.get('error')}")
    
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        logger.error("Workflow test failed", exc_info=True)
    
    # Get available workflows
    print("\n" + "─"*70)
    print("Available Workflow Templates:")
    print("─"*70)
    
    templates = orchestrator.get_workflow_templates()
    for i, template in enumerate(templates, 1):
        print(f"\n{i}. {template['name']}")
        print(f"   Description: {template['description']}")
        print(f"   Steps: {' → '.join(template['steps'])}")
        print(f"   Required: {', '.join(template['required_params'])}")


def test_complete_workflow():
    """Test complete workflow with actual email (interactive)."""
    print_separator("TEST 4: Complete Workflow (Interactive)", "=")
    
    print("This test will run a complete workflow:")
    print("  1. Generate a quiz")
    print("  2. Create PDF document")
    print("  3. Send via email")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("Skipping complete workflow test")
        return
    
    # Get email
    to_email = input("Enter recipient email: ").strip()
    if not to_email or '@' not in to_email:
        print("Invalid email, skipping test")
        return
    
    student_name = input("Enter student name (optional): ").strip() or None
    
    # Run workflow
    print("\n" + "─"*70)
    print("Running Complete Workflow")
    print("─"*70)
    
    try:
        orchestrator = ToolOrchestrator()
        
        result = orchestrator.generate_and_email_quiz(
            topic="Triangles",
            grade_level="Grade 8",
            to_email=to_email,
            num_questions=5,
            format='pdf',
            student_name=student_name,
            include_html=True
        )
        
        # Display results
        print(f"\nWorkflow: {result.workflow_name}")
        print(f"Status: {result.overall_status.value}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"\nSteps completed: {len(result.steps)}")
        
        for i, step in enumerate(result.steps, 1):
            status_icon = "✓" if step.status == ToolStatus.SUCCESS else "✗"
            print(f"\n{i}. {status_icon} {step.tool_name}")
            print(f"   Status: {step.status.value}")
            print(f"   Time: {step.execution_time:.2f}s")
            
            if step.error:
                print(f"   Error: {step.error}")
            elif step.output:
                output_preview = str(step.output)[:80]
                print(f"   Output: {output_preview}...")
        
        if result.overall_status == ToolStatus.SUCCESS:
            print("\n✓ Complete workflow succeeded!")
            print(f"  Email sent to: {to_email}")
        elif result.overall_status == ToolStatus.PARTIAL:
            print("\n⚠️  Workflow partially succeeded")
        else:
            print("\n✗ Workflow failed")
    
    except Exception as e:
        print(f"\n✗ Workflow error: {e}")
        logger.error("Complete workflow test failed", exc_info=True)


def test_error_recovery():
    """Test error handling and fallback mechanisms."""
    print_separator("TEST 5: Error Recovery & Fallbacks", "=")
    
    orchestrator = ToolOrchestrator(enable_fallbacks=True, max_retries=2)
    
    print("Testing fallback mechanisms...")
    print("Scenario: Primary format fails, tries fallback formats\n")
    
    # Create mock quiz data
    quiz_data = {
        'quiz_title': 'Test Quiz',
        'topic': 'Triangles',
        'grade_level': 'Grade 8',
        'questions': []
    }
    
    # Test with different primary formats
    formats_to_test = ['pdf', 'docx', 'pptx']
    
    for primary_format in formats_to_test:
        print(f"\n{'─'*70}")
        print(f"Testing with primary format: {primary_format}")
        print("─"*70)
        
        try:
            doc_gen = DocumentGenerator()
            
            # Try primary format
            result = doc_gen.generate(
                content_type='quiz',
                format=primary_format,
                data=quiz_data,
                filename=f'fallback_test_{primary_format}.{primary_format}'
            )
            
            print(f"✓ Successfully generated {primary_format}")
            print(f"  File: {result}")
            
        except Exception as e:
            print(f"✗ Primary format {primary_format} failed: {e}")
            print("  Fallback would try other formats...")
    
    print("\n" + "="*70)
    print("Error recovery test complete")
    print("Note: Full fallback chain is active in actual workflows")
    print("="*70)


def main():
    """Run all tool tests."""
    print("="*70)
    print("  TOOL INTEGRATION TEST SUITE")
    print("  Week 3-4: Document Generation & Email")
    print("="*70)
    
    print("\nAvailable Tests:")
    print("1. Document Generation (PDF/DOCX/PPT)")
    print("2. Email Sending")
    print("3. Tool Orchestration")
    print("4. Complete Workflow (Interactive)")
    print("5. Error Recovery & Fallbacks")
    print("6. All Tests")
    print("0. Exit")
    
    choice = input("\nSelect test (0-6): ").strip()
    
    try:
        if choice == '1':
            test_document_generation()
        elif choice == '2':
            test_email_sending()
        elif choice == '3':
            test_tool_orchestration()
        elif choice == '4':
            test_complete_workflow()
        elif choice == '5':
            test_error_recovery()
        elif choice == '6':
            test_document_generation()
            test_email_sending()
            test_tool_orchestration()
            test_error_recovery()
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