#!/usr/bin/env python
"""
Tool Orchestrator
Manages and coordinates all tools (document generation, email, RAG)
Provides unified interface and error recovery
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import time
from dataclasses import dataclass, field

from .document_generator import DocumentGenerator, DocumentGenerationError
from .email_sender import EmailSender, EmailError, create_html_quiz_email, create_html_report_email
from ..llm.rag_llm_pipeline import GeometryTutorPipeline

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Available tool types."""
    DOCUMENT_GENERATION = "document_generation"
    EMAIL = "email"
    RAG_RETRIEVAL = "rag_retrieval"
    LLM_GENERATION = "llm_generation"


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tool_name': self.tool_name,
            'status': self.status.value,
            'output': self.output,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class WorkflowResult:
    """Result from multi-step workflow execution."""
    workflow_name: str
    overall_status: ToolStatus
    steps: List[ToolResult] = field(default_factory=list)
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'workflow_name': self.workflow_name,
            'overall_status': self.overall_status.value,
            'total_time': self.total_time,
            'steps': [step.to_dict() for step in self.steps],
            'success_count': sum(1 for s in self.steps if s.status == ToolStatus.SUCCESS),
            'failed_count': sum(1 for s in self.steps if s.status == ToolStatus.FAILED),
            'total_steps': len(self.steps)
        }


class ToolOrchestrator:
    """
    Orchestrates multiple tools to fulfill complex user requests.
    Handles error recovery, retries, and fallback strategies.
    """
    
    def __init__(
        self,
        tutor_pipeline: Optional[GeometryTutorPipeline] = None,
        document_generator: Optional[DocumentGenerator] = None,
        email_sender: Optional[EmailSender] = None,
        max_retries: int = 2,
        enable_fallbacks: bool = True
    ):
        """
        Initialize tool orchestrator.
        
        Args:
            tutor_pipeline: RAG+LLM pipeline (created if None)
            document_generator: Document generator (created if None)
            email_sender: Email sender (created if None)
            max_retries: Maximum retry attempts for failed tools
            enable_fallbacks: Enable fallback strategies
        """
        # Initialize tools
        self.tutor = tutor_pipeline or GeometryTutorPipeline()
        self.doc_gen = document_generator or DocumentGenerator()
        self.email = email_sender or EmailSender()
        
        self.max_retries = max_retries
        self.enable_fallbacks = enable_fallbacks
        
        # Track tool availability
        self.available_tools = {
            ToolType.RAG_RETRIEVAL: True,
            ToolType.LLM_GENERATION: True,
            ToolType.DOCUMENT_GENERATION: any(self.doc_gen.get_supported_formats().values()),
            ToolType.EMAIL: bool(self.email.username and self.email.password)
        }
        
        logger.info("Tool Orchestrator initialized")
        logger.info(f"  RAG Retrieval: {'✓' if self.available_tools[ToolType.RAG_RETRIEVAL] else '✗'}")
        logger.info(f"  LLM Generation: {'✓' if self.available_tools[ToolType.LLM_GENERATION] else '✗'}")
        logger.info(f"  Document Gen: {'✓' if self.available_tools[ToolType.DOCUMENT_GENERATION] else '✗'}")
        logger.info(f"  Email: {'✓' if self.available_tools[ToolType.EMAIL] else '✗'}")
    
    # ========== Individual Tool Execution ==========
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_func: Callable,
        *args,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool with error handling and retry logic.
        
        Args:
            tool_name: Name of the tool
            tool_func: Tool function to execute
            *args, **kwargs: Arguments for tool function
        
        Returns:
            ToolResult
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Executing tool: {tool_name} (attempt {attempt + 1})")
                
                output = tool_func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Check if tool returned error status
                if isinstance(output, dict) and not output.get('success', True):
                    raise Exception(output.get('error', 'Tool execution failed'))
                
                logger.info(f"✓ Tool {tool_name} succeeded in {elapsed:.2f}s")
                
                return ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.SUCCESS,
                    output=output,
                    execution_time=elapsed
                )
                
            except Exception as e:
                logger.error(f"Tool {tool_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying tool {tool_name}...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    elapsed = time.time() - start_time
                    return ToolResult(
                        tool_name=tool_name,
                        status=ToolStatus.FAILED,
                        error=str(e),
                        execution_time=elapsed
                    )
    
    # ========== Workflow Definitions ==========
    
    def generate_and_email_quiz(
        self,
        topic: str,
        grade_level: str,
        to_email: str,
        num_questions: int = 5,
        format: str = 'pdf',
        student_name: Optional[str] = None,
        include_html: bool = True
    ) -> WorkflowResult:
        """
        Complete workflow: Generate quiz → Create document → Email to student.
        
        Args:
            topic: Quiz topic
            grade_level: Target grade
            to_email: Student email
            num_questions: Number of questions
            format: Document format (pdf/docx/pptx)
            student_name: Student name for personalization
            include_html: Use HTML email template
        
        Returns:
            WorkflowResult with all steps
        """
        workflow_name = f"generate_and_email_quiz_{topic}"
        start_time = time.time()
        steps = []
        
        logger.info(f"Starting workflow: {workflow_name}")
        
        # Step 1: Generate quiz content
        quiz_result = self._execute_tool(
            "generate_quiz",
            self.tutor.generate_quiz,
            topic=topic,
            grade_level=grade_level,
            num_questions=num_questions,
            use_structured_output=True
        )
        steps.append(quiz_result)
        
        if quiz_result.status != ToolStatus.SUCCESS:
            logger.error("Quiz generation failed, aborting workflow")
            return WorkflowResult(
                workflow_name=workflow_name,
                overall_status=ToolStatus.FAILED,
                steps=steps,
                total_time=time.time() - start_time
            )
        
        quiz_data = quiz_result.output
        
        # Step 2: Generate document (with fallback)
        doc_formats_to_try = [format]
        if self.enable_fallbacks:
            # Add fallback formats
            if format == 'pdf':
                doc_formats_to_try.extend(['docx', 'pptx'])
            elif format == 'docx':
                doc_formats_to_try.extend(['pdf', 'pptx'])
            elif format in ['ppt', 'pptx']:
                doc_formats_to_try.extend(['pdf', 'docx'])
        
        doc_result = None
        for fmt in doc_formats_to_try:
            doc_result = self._execute_tool(
                f"generate_document_{fmt}",
                self.doc_gen.generate,
                content_type='quiz',
                format=fmt,
                data=quiz_data,
                include_answers=True
            )
            
            if doc_result.status == ToolStatus.SUCCESS:
                logger.info(f"✓ Document generated in {fmt} format")
                break
            else:
                logger.warning(f"Failed to generate {fmt}, trying fallback...")
        
        steps.append(doc_result)
        
        if doc_result.status != ToolStatus.SUCCESS:
            logger.error("All document generation attempts failed")
            return WorkflowResult(
                workflow_name=workflow_name,
                overall_status=ToolStatus.FAILED,
                steps=steps,
                total_time=time.time() - start_time
            )
        
        document_path = doc_result.output
        
        # Step 3: Send email
        if include_html:
            html_body = create_html_quiz_email(quiz_data, student_name)
            email_result = self._execute_tool(
                "send_email",
                self.email.send_email,
                to_email=to_email,
                subject=f"Geometry Quiz: {topic}",
                body=html_body,
                attachments=[document_path],
                html=True
            )
        else:
            email_result = self._execute_tool(
                "send_quiz_email",
                self.email.send_quiz,
                to_email=to_email,
                quiz_data=quiz_data,
                attachments=[document_path],
                student_name=student_name
            )
        
        steps.append(email_result)
        
        # Determine overall status
        if all(s.status == ToolStatus.SUCCESS for s in steps):
            overall_status = ToolStatus.SUCCESS
        elif any(s.status == ToolStatus.SUCCESS for s in steps):
            overall_status = ToolStatus.PARTIAL
        else:
            overall_status = ToolStatus.FAILED
        
        total_time = time.time() - start_time
        
        logger.info(f"Workflow {workflow_name} completed: {overall_status.value} ({total_time:.2f}s)")
        
        return WorkflowResult(
            workflow_name=workflow_name,
            overall_status=overall_status,
            steps=steps,
            total_time=total_time
        )
    
    def generate_and_email_explanation(
        self,
        concept: str,
        grade_level: str,
        to_email: str,
        explanation_type: str = "step-by-step",
        format: str = 'pdf',
        student_name: Optional[str] = None
    ) -> WorkflowResult:
        """
        Complete workflow: Generate explanation → Create document → Email to student.
        
        Args:
            concept: Concept to explain
            grade_level: Target grade
            to_email: Student email
            explanation_type: Type of explanation
            format: Document format
            student_name: Student name
        
        Returns:
            WorkflowResult
        """
        workflow_name = f"generate_and_email_explanation_{concept}"
        start_time = time.time()
        steps = []
        
        logger.info(f"Starting workflow: {workflow_name}")
        
        # Step 1: Generate explanation
        explain_result = self._execute_tool(
            "generate_explanation",
            self.tutor.explain_concept,
            concept=concept,
            grade_level=grade_level,
            explanation_type=explanation_type,
            include_examples=True
        )
        steps.append(explain_result)
        
        if explain_result.status != ToolStatus.SUCCESS:
            return WorkflowResult(
                workflow_name=workflow_name,
                overall_status=ToolStatus.FAILED,
                steps=steps,
                total_time=time.time() - start_time
            )
        
        explanation_data = explain_result.output
        
        # Step 2: Generate document (with fallback)
        doc_formats = [format, 'pdf', 'docx'] if self.enable_fallbacks else [format]
        
        doc_result = None
        for fmt in doc_formats:
            doc_result = self._execute_tool(
                f"generate_document_{fmt}",
                self.doc_gen.generate,
                content_type='explanation',
                format=fmt,
                data=explanation_data
            )
            
            if doc_result.status == ToolStatus.SUCCESS:
                break
        
        steps.append(doc_result)
        
        if doc_result.status != ToolStatus.SUCCESS:
            return WorkflowResult(
                workflow_name=workflow_name,
                overall_status=ToolStatus.FAILED,
                steps=steps,
                total_time=time.time() - start_time
            )
        
        document_path = doc_result.output
        
        # Step 3: Send email
        explanation_preview = explanation_data.get('explanation', '')[:300]
        
        email_result = self._execute_tool(
            "send_explanation_email",
            self.email.send_explanation,
            to_email=to_email,
            concept=concept,
            explanation_preview=explanation_preview,
            attachments=[document_path],
            student_name=student_name
        )
        
        steps.append(email_result)
        
        # Determine overall status
        if all(s.status == ToolStatus.SUCCESS for s in steps):
            overall_status = ToolStatus.SUCCESS
        elif any(s.status == ToolStatus.SUCCESS for s in steps):
            overall_status = ToolStatus.PARTIAL
        else:
            overall_status = ToolStatus.FAILED
        
        total_time = time.time() - start_time
        
        logger.info(f"Workflow completed: {overall_status.value} ({total_time:.2f}s)")
        
        return WorkflowResult(
            workflow_name=workflow_name,
            overall_status=overall_status,
            steps=steps,
            total_time=total_time
        )
    
    def batch_quiz_generation(
        self,
        topics: List[str],
        grade_level: str,
        format: str = 'pdf',
        output_dir: Optional[str] = None
    ) -> WorkflowResult:
        """
        Generate multiple quizzes in batch.
        
        Args:
            topics: List of topics
            grade_level: Target grade
            format: Document format
            output_dir: Optional output directory
        
        Returns:
            WorkflowResult
        """
        workflow_name = "batch_quiz_generation"
        start_time = time.time()
        steps = []
        
        logger.info(f"Starting batch quiz generation: {len(topics)} topics")
        
        for topic in topics:
            # Generate quiz
            quiz_result = self._execute_tool(
                f"generate_quiz_{topic}",
                self.tutor.generate_quiz,
                topic=topic,
                grade_level=grade_level,
                num_questions=5
            )
            steps.append(quiz_result)
            
            if quiz_result.status == ToolStatus.SUCCESS:
                # Generate document
                doc_result = self._execute_tool(
                    f"generate_doc_{topic}",
                    self.doc_gen.generate,
                    content_type='quiz',
                    format=format,
                    data=quiz_result.output
                )
                steps.append(doc_result)
        
        # Determine overall status
        success_count = sum(1 for s in steps if s.status == ToolStatus.SUCCESS)
        if success_count == len(steps):
            overall_status = ToolStatus.SUCCESS
        elif success_count > 0:
            overall_status = ToolStatus.PARTIAL
        else:
            overall_status = ToolStatus.FAILED
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch generation complete: {success_count}/{len(steps)} succeeded")
        
        return WorkflowResult(
            workflow_name=workflow_name,
            overall_status=overall_status,
            steps=steps,
            total_time=total_time
        )
    
    # ========== Utility Methods ==========
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of all tools."""
        return {
            'available_tools': {k.value: v for k, v in self.available_tools.items()},
            'document_formats': self.doc_gen.get_supported_formats(),
            'email_configured': bool(self.email.username and self.email.password),
            'max_retries': self.max_retries,
            'fallbacks_enabled': self.enable_fallbacks
        }
    
    def test_all_tools(self) -> Dict[str, bool]:
        """Test all tools and return status."""
        results = {}
        
        # Test email
        if self.email.username and self.email.password:
            results['email'] = self.email.test_connection()
        else:
            results['email'] = False
        
        # Test document generation (try to generate a simple doc)
        try:
            test_data = {
                'title': 'Test',
                'content': 'Test document',
                'metadata': {}
            }
            
            formats = self.doc_gen.get_supported_formats()
            results['pdf'] = formats.get('pdf', False)
            results['docx'] = formats.get('docx', False)
            results['pptx'] = formats.get('pptx', False)
        
        except Exception as e:
            logger.error(f"Document generation test failed: {e}")
            results['pdf'] = False
            results['docx'] = False
            results['pptx'] = False
        
        # Test RAG pipeline
        try:
            stats = self.tutor.get_statistics()
            results['rag'] = stats.get('status') == 'healthy'
        except Exception:
            results['rag'] = False
        
        return results
    
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates."""
        return [
            {
                'name': 'generate_and_email_quiz',
                'description': 'Generate quiz, create document, and email to student',
                'steps': ['Generate Quiz', 'Create Document', 'Send Email'],
                'required_params': ['topic', 'grade_level', 'to_email']
            },
            {
                'name': 'generate_and_email_explanation',
                'description': 'Generate explanation, create document, and email to student',
                'steps': ['Generate Explanation', 'Create Document', 'Send Email'],
                'required_params': ['concept', 'grade_level', 'to_email']
            },
            {
                'name': 'batch_quiz_generation',
                'description': 'Generate multiple quizzes in batch',
                'steps': ['Generate Quiz (x N)', 'Create Documents (x N)'],
                'required_params': ['topics', 'grade_level']
            }
        ]