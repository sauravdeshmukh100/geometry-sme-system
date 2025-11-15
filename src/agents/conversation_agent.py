#!/usr/bin/env python
"""
Conversation Agent & Workflow Planner
Handles multi-turn conversations and plans complex workflows
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from .task_router import TaskRouter, TaskType, ConversationMemory
from ..llm.rag_llm_pipeline import GeometryTutorPipeline
from ..tools.tool_orchestrator import ToolOrchestrator, WorkflowResult

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Workflow execution steps."""
    UNDERSTAND = "understand"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    VALIDATE = "validate"
    DELIVER = "deliver"


@dataclass
class WorkflowPlan:
    """Plan for executing a workflow."""
    workflow_name: str
    steps: List[WorkflowStep]
    parameters: Dict[str, Any]
    estimated_time: float = 0.0
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'workflow_name': self.workflow_name,
            'steps': [step.value for step in self.steps],
            'parameters': self.parameters,
            'estimated_time': self.estimated_time,
            'requires_confirmation': self.requires_confirmation
        }


class ConversationAgent:
    """
    Main conversation agent that handles multi-turn interactions.
    Coordinates between task routing, memory, and workflow execution.
    """
    
    def __init__(
        self,
        tutor_pipeline: Optional[GeometryTutorPipeline] = None,
        tool_orchestrator: Optional[ToolOrchestrator] = None
    ):
        """
        Initialize conversation agent.
        
        Args:
            tutor_pipeline: RAG+LLM pipeline
            tool_orchestrator: Tool orchestration system
        """
        # Initialize components
        self.tutor = tutor_pipeline or GeometryTutorPipeline()
        self.orchestrator = tool_orchestrator or ToolOrchestrator()
        self.router = TaskRouter()
        self.memory = ConversationMemory(max_history=10)
        
        # State
        self.current_workflow = None
        self.pending_confirmations = []
        
        logger.info("Conversation Agent initialized")
    
    def process_message(
        self, 
        message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
        Args:
            message: User's message
            user_id: Optional user identifier
        
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Processing message: {message[:50]}...")
        
        # Resolve references using memory
        resolved_message = self.memory.resolve_references(message)
        
        # Get conversation context
        context = self.memory.get_context(num_turns=3)
        
        # Route the message
        routing = self.router.route_with_disambiguation(resolved_message, context)
        
        # Check if clarification needed
        if routing.get('needs_clarification'):
            response = {
                'type': 'clarification',
                'message': "I'm not sure what you'd like me to do. Could you clarify?",
                'options': routing.get('clarification_options', []),
                'routing': routing
            }
            return response
        
        # Execute based on task type
        task_type = routing['task_type']
        
        try:
            if task_type == TaskType.GREETING.value:
                response = self._handle_greeting(message, context)
            
            elif task_type == TaskType.HELP.value:
                response = self._handle_help(message)
            
            elif task_type == TaskType.QUESTION_ANSWER.value:
                response = self._handle_question(message, routing, context)
            
            elif task_type == TaskType.QUIZ_GENERATION.value:
                response = self._handle_quiz_request(message, routing, context)
            
            elif task_type == TaskType.CONCEPT_EXPLANATION.value:
                response = self._handle_explanation_request(message, routing, context)
            
            elif task_type == TaskType.WORKFLOW_COMPLEX.value:
                response = self._handle_complex_workflow(message, routing, context)
            
            else:
                response = self._handle_unknown(message)
            
            # Add to memory
            self.memory.add_turn(
                user_message=message,
                assistant_response=response.get('message', ''),
                task_type=task_type,
                entities=routing.get('entities')
            )
            
            # Add metadata
            response['routing'] = routing
            response['conversation_length'] = len(self.memory.history)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f"I encountered an error: {str(e)}",
                'error': str(e)
            }
    
    def _handle_greeting(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting messages."""
        greetings = [
            "Hello! I'm your Geometry tutor. I can help you with:",
            "• Answering geometry questions",
            "• Generating quizzes and practice tests",
            "• Explaining concepts step-by-step",
            "• Creating study materials",
            "\nWhat would you like to learn today?"
        ]
        
        return {
            'type': 'greeting',
            'message': '\n'.join(greetings)
        }
    
    def _handle_help(self, message: str) -> Dict[str, Any]:
        """Handle help requests."""
        help_text = """I can help you with geometry in several ways:

**Ask Questions:**
- "What is the Pythagorean theorem?"
- "How do I calculate the area of a circle?"
- "Explain properties of triangles"

**Generate Quizzes:**
- "Create a quiz on triangles for Grade 8"
- "Generate 10 questions about circles"
- "Make a test on quadrilaterals and email it to student@school.edu"

**Get Explanations:**
- "Explain the Pythagorean theorem step-by-step"
- "Describe types of triangles with examples"
- "What are the properties of parallel lines?"

**Create Documents:**
- "Generate a PDF quiz on circles"
- "Create a PowerPoint presentation on triangles"
- "Make a study guide about angles"

Just ask naturally, and I'll understand what you need!"""
        
        return {
            'type': 'help',
            'message': help_text
        }
    
    def _handle_question(
        self, 
        message: str, 
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle question answering."""
        params = routing['parameters']
        
        # Get grade level from context or params
        grade_level = params.get('grade_level') or context.get('entities', {}).get('grade_level')
        
        # Answer the question
        result = self.tutor.answer_question(
            query=message,
            grade_level=grade_level,
            top_k=params.get('top_k', 5)
        )
        
        if result.get('success'):
            return {
                'type': 'answer',
                'message': result['answer'],
                'sources': result.get('sources', []),
                'metadata': result.get('retrieval_metadata', {})
            }
        else:
            return {
                'type': 'error',
                'message': "I couldn't find an answer to your question. Could you rephrase it?",
                'error': result.get('error')
            }
    
    def _handle_quiz_request(
        self,
        message: str,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle quiz generation requests."""
        params = routing['parameters']
        entities = routing['entities']
        
        # Check if email is requested
        email = entities.get('email')
        
        if email:
            # Need confirmation for email
            return {
                'type': 'confirmation_needed',
                'message': f"I'll generate a quiz and email it to {email}. Confirm?",
                'pending_action': {
                    'action': 'generate_and_email_quiz',
                    'params': params
                }
            }
        else:
            # Just generate quiz
            topic = entities.get('topic', 'geometry')
            grade = params.get('grade_level', 'Grade 8')
            num_q = params.get('num_questions', 5)
            
            result = self.tutor.generate_quiz(
                topic=topic,
                grade_level=grade,
                num_questions=num_q
            )
            
            if result.get('success'):
                # Also generate document
                from ..tools.document_generator import DocumentGenerator
                doc_gen = DocumentGenerator()
                
                try:
                    doc_path = doc_gen.generate(
                        content_type='quiz',
                        format=params.get('format', 'pdf'),
                        data=result
                    )
                    
                    return {
                        'type': 'quiz_generated',
                        'message': f"I've generated a {num_q}-question quiz on {topic} for {grade}.",
                        'quiz_preview': result.get('quiz', '')[:300] + "...",
                        'document_path': doc_path,
                        'metadata': result.get('metadata', {})
                    }
                except Exception as e:
                    logger.error(f"Document generation failed: {e}")
                    return {
                        'type': 'quiz_generated',
                        'message': f"I've generated the quiz content, but couldn't create the document.",
                        'quiz_preview': result.get('quiz', '')[:300] + "...",
                        'error': str(e)
                    }
            else:
                return {
                    'type': 'error',
                    'message': "I couldn't generate the quiz. Please try again.",
                    'error': result.get('error')
                }
    
    def _handle_explanation_request(
        self,
        message: str,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle concept explanation requests."""
        params = routing['parameters']
        entities = routing['entities']
        
        # Extract concept (fallback to message if not found)
        concept = entities.get('topic', message)
        grade = params.get('grade_level', 'Grade 8')
        exp_type = params.get('explanation_type', 'step-by-step')
        
        result = self.tutor.explain_concept(
            concept=concept,
            grade_level=grade,
            explanation_type=exp_type,
            include_examples=True
        )
        
        if result.get('success'):
            explanation = result.get('explanation', '')
            
            return {
                'type': 'explanation',
                'message': explanation,
                'concept': concept,
                'metadata': result.get('metadata', {})
            }
        else:
            return {
                'type': 'error',
                'message': f"I couldn't explain {concept}. Could you rephrase?",
                'error': result.get('error')
            }
    
    def _handle_complex_workflow(
        self,
        message: str,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle complex multi-step workflows."""
        # Plan the workflow
        plan = self._plan_workflow(message, routing, context)
        
        return {
            'type': 'workflow_planned',
            'message': f"I'll execute this workflow: {plan.workflow_name}",
            'plan': plan.to_dict(),
            'requires_confirmation': plan.requires_confirmation
        }
    
    def _handle_unknown(self, message: str) -> Dict[str, Any]:
        """Handle unknown task types."""
        return {
            'type': 'unknown',
            'message': "I'm not sure how to help with that. Could you try rephrasing or type 'help' to see what I can do?"
        }
    
    def _plan_workflow(
        self,
        message: str,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> WorkflowPlan:
        """
        Plan a multi-step workflow based on user request.
        
        Args:
            message: User's message
            routing: Routing information
            context: Conversation context
        
        Returns:
            WorkflowPlan
        """
        # Analyze the request to determine steps
        steps = [WorkflowStep.UNDERSTAND]
        
        # If requires context retrieval
        if routing.get('requires_context'):
            steps.append(WorkflowStep.RETRIEVE)
        
        # Generation step
        steps.append(WorkflowStep.GENERATE)
        
        # If document or email involved
        if 'format' in routing.get('entities', {}) or 'email' in routing.get('entities', {}):
            steps.append(WorkflowStep.VALIDATE)
            steps.append(WorkflowStep.DELIVER)
        
        # Estimate time (rough)
        estimated_time = len(steps) * 3.0  # ~3 seconds per step
        
        # Requires confirmation if email is involved
        requires_confirmation = 'email' in routing.get('entities', {})
        
        return WorkflowPlan(
            workflow_name=routing.get('suggested_workflow', 'custom'),
            steps=steps,
            parameters=routing.get('parameters', {}),
            estimated_time=estimated_time,
            requires_confirmation=requires_confirmation
        )
    
    def execute_workflow(self, plan: WorkflowPlan) -> WorkflowResult:
        """
        Execute a planned workflow.
        
        Args:
            plan: Workflow plan to execute
        
        Returns:
            WorkflowResult
        """
        workflow_name = plan.workflow_name
        params = plan.parameters
        
        logger.info(f"Executing workflow: {workflow_name}")
        
        # Delegate to orchestrator based on workflow name
        if workflow_name == 'generate_and_email_quiz':
            return self.orchestrator.generate_and_email_quiz(
                topic=params.get('topic', 'geometry'),
                grade_level=params.get('grade_level', 'Grade 8'),
                to_email=params.get('email'),
                num_questions=params.get('num_questions', 5),
                format=params.get('format', 'pdf')
            )
        
        elif workflow_name == 'generate_quiz':
            # Simplified workflow without email
            result = self.tutor.generate_quiz(
                topic=params.get('topic', 'geometry'),
                grade_level=params.get('grade_level', 'Grade 8'),
                num_questions=params.get('num_questions', 5)
            )
            
            # Wrap in WorkflowResult format
            from ..tools.tool_orchestrator import ToolResult, ToolStatus
            
            return WorkflowResult(
                workflow_name='generate_quiz',
                overall_status=ToolStatus.SUCCESS if result.get('success') else ToolStatus.FAILED,
                steps=[
                    ToolResult(
                        tool_name='generate_quiz',
                        status=ToolStatus.SUCCESS if result.get('success') else ToolStatus.FAILED,
                        output=result
                    )
                ]
            )
        
        else:
            raise ValueError(f"Unknown workflow: {workflow_name}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and available features."""
        return {
            'tasks': [
                {
                    'type': 'question_answer',
                    'description': 'Answer geometry questions with context',
                    'example': 'What is the Pythagorean theorem?'
                },
                {
                    'type': 'quiz_generation',
                    'description': 'Generate quizzes on any geometry topic',
                    'example': 'Create a 5-question quiz on triangles for Grade 8'
                },
                {
                    'type': 'concept_explanation',
                    'description': 'Explain concepts with examples',
                    'example': 'Explain properties of circles step-by-step'
                },
                {
                    'type': 'document_generation',
                    'description': 'Create PDF/DOCX/PPT documents',
                    'example': 'Generate a PDF study guide on angles'
                },
                {
                    'type': 'email_delivery',
                    'description': 'Email generated content to students',
                    'example': 'Generate a quiz and email it to student@school.edu'
                }
            ],
            'features': {
                'conversation_memory': True,
                'context_retrieval': True,
                'multi_step_workflows': True,
                'error_recovery': True,
                'document_formats': ['pdf', 'docx', 'pptx']
            },
            'supported_grades': ['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']
        }
    
    def reset_conversation(self):
        """Reset conversation memory and state."""
        self.memory.clear()
        self.current_workflow = None
        self.pending_confirmations = []
        logger.info("Conversation reset")