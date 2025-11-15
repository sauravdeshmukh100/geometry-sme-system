#!/usr/bin/env python
"""
Task Router Agent
Classifies user queries and routes to appropriate tools/workflows
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import re

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks the system can handle."""
    QUESTION_ANSWER = "question_answer"
    QUIZ_GENERATION = "quiz_generation"
    CONCEPT_EXPLANATION = "concept_explanation"
    DOCUMENT_GENERATION = "document_generation"
    EMAIL_SEND = "email_send"
    WORKFLOW_COMPLEX = "workflow_complex"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"


class IntentClassifier:
    """
    Classifies user intent using rule-based and pattern matching.
    Can be extended with ML-based classification.
    """
    
    def __init__(self):
        """Initialize intent classifier with patterns."""
        # Intent patterns (keyword-based)
        self.patterns = {
            TaskType.QUIZ_GENERATION: [
                r'\b(quiz|test|exam|assessment|questions?)\b',
                r'\b(generate|create|make)\s+(a\s+)?(quiz|test)',
                r'\b(give|show)\s+me\s+(a\s+)?(quiz|test)',
            ],
            TaskType.CONCEPT_EXPLANATION: [
                r'\b(explain|describe|what\s+is|define|tell\s+me\s+about)\b',
                r'\b(how\s+does|how\s+do|why\s+is)\b',
                r'\b(concept|theory|theorem|principle)\b',
                r'\b(step\s+by\s+step|detailed)\s+(explanation|guide)',
            ],
            TaskType.QUESTION_ANSWER: [
                r'\b(what|when|where|who|which|how)\b',
                r'\?$',  # Ends with question mark
                r'\b(can\s+you|could\s+you|will\s+you)\b',
                r'\b(solve|calculate|find|compute)\b',
            ],
            TaskType.DOCUMENT_GENERATION: [
                r'\b(generate|create|make)\s+(a\s+)?(document|pdf|docx|report|ppt)',
                r'\b(export|download|save)\s+(as|to)\b',
                r'\b(send|email)\s+me\s+(a|the)\s+(document|pdf)',
            ],
            TaskType.EMAIL_SEND: [
                r'\b(email|send|mail)\b',
                r'\b(send\s+to|email\s+to)\b',
                r'@',  # Email address present
            ],
            TaskType.GREETING: [
                r'\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b',
                r'\b(how\s+are\s+you|what\'?s\s+up)\b',
            ],
            TaskType.HELP: [
                r'\b(help|assist|support|guide)\b',
                r'\b(what\s+can\s+you|how\s+to\s+use)\b',
                r'\b(show\s+me|tell\s+me)\s+.*\s+(options|features|capabilities)\b',
            ],
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            task_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for task_type, patterns in self.patterns.items()
        }
        
        # Keywords for specific tasks
        self.quiz_keywords = ['quiz', 'test', 'exam', 'questions', 'assessment', 'mcq']
        self.explain_keywords = ['explain', 'describe', 'what is', 'how does', 'why', 'concept']
        self.document_keywords = ['pdf', 'docx', 'ppt', 'document', 'report', 'presentation']
        self.email_keywords = ['email', 'send', 'mail', 'deliver']
        
        logger.info("Intent Classifier initialized")
    
    def classify(self, query: str) -> Tuple[TaskType, float]:
        """
        Classify user query into task type.
        
        Args:
            query: User's input query
        
        Returns:
            Tuple of (TaskType, confidence_score)
        """
        query_lower = query.lower().strip()
        
        # Empty query
        if not query_lower:
            return TaskType.UNKNOWN, 0.0
        
        # Check each pattern
        scores = {}
        
        for task_type, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(query_lower))
            if matches > 0:
                # Score based on number of pattern matches
                scores[task_type] = matches / len(patterns)
        
        # If no patterns matched, use default Q&A
        if not scores:
            # Check if it looks like a question
            if '?' in query or any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
                return TaskType.QUESTION_ANSWER, 0.6
            return TaskType.UNKNOWN, 0.3
        
        # Get task type with highest score
        best_task = max(scores.items(), key=lambda x: x[1])
        
        return best_task[0], best_task[1]
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from query (topic, grade, email, etc.).
        
        Args:
            query: User's input query
        
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Extract grade level
        grade_pattern = r'\b(grade|class|standard)\s*(\d{1,2})\b'
        grade_match = re.search(grade_pattern, query, re.IGNORECASE)
        if grade_match:
            entities['grade_level'] = f"Grade {grade_match.group(2)}"
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, query)
        if email_match:
            entities['email'] = email_match.group(0)
        
        # Extract number of questions
        num_pattern = r'\b(\d+)\s+(questions?|problems?|items?)\b'
        num_match = re.search(num_pattern, query, re.IGNORECASE)
        if num_match:
            entities['num_questions'] = int(num_match.group(1))
        
        # Extract document format
        format_pattern = r'\b(pdf|docx|ppt|pptx|word|powerpoint)\b'
        format_match = re.search(format_pattern, query, re.IGNORECASE)
        if format_match:
            fmt = format_match.group(1).lower()
            if fmt == 'word':
                fmt = 'docx'
            elif fmt == 'powerpoint':
                fmt = 'pptx'
            entities['format'] = fmt
        
        # Extract topic (heuristic: capitalized words or words after "about/on")
        topic_pattern = r'\b(about|on|regarding|concerning)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        topic_match = re.search(topic_pattern, query)
        if topic_match:
            entities['topic'] = topic_match.group(2)
        
        return entities


class TaskRouter:
    """
    Routes tasks to appropriate handlers based on intent classification.
    Manages task execution and response generation.
    """
    
    def __init__(self):
        """Initialize task router."""
        self.classifier = IntentClassifier()
        self.task_handlers = {}
        self.conversation_history = []
        
        logger.info("Task Router initialized")
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route query to appropriate task handler.
        
        Args:
            query: User's input query
            context: Optional conversation context
        
        Returns:
            Routing decision with task type, confidence, and parameters
        """
        # Classify intent
        task_type, confidence = self.classifier.classify(query)
        
        # Extract entities
        entities = self.classifier.extract_entities(query)
        
        # Build routing decision
        routing = {
            'query': query,
            'task_type': task_type.value,
            'confidence': confidence,
            'entities': entities,
            'requires_context': self._requires_retrieval(task_type),
            'suggested_workflow': self._suggest_workflow(task_type, entities),
            'parameters': self._build_parameters(task_type, entities, context)
        }
        
        logger.info(f"Routed query to: {task_type.value} (confidence: {confidence:.2f})")
        
        return routing
    
    def route_with_disambiguation(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route with disambiguation for ambiguous queries.
        
        Args:
            query: User's input query
            context: Optional conversation context
        
        Returns:
            Routing decision, possibly with disambiguation options
        """
        routing = self.route(query, context)
        
        # If confidence is low, provide alternatives
        if routing['confidence'] < 0.5:
            routing['needs_clarification'] = True
            routing['clarification_options'] = self._generate_clarification_options(query)
        else:
            routing['needs_clarification'] = False
        
        return routing
    
    def _requires_retrieval(self, task_type: TaskType) -> bool:
        """Check if task requires RAG retrieval."""
        retrieval_tasks = {
            TaskType.QUESTION_ANSWER,
            TaskType.QUIZ_GENERATION,
            TaskType.CONCEPT_EXPLANATION
        }
        return task_type in retrieval_tasks
    
    def _suggest_workflow(self, task_type: TaskType, entities: Dict[str, Any]) -> str:
        """
        Suggest appropriate workflow based on task type and entities.
        
        Args:
            task_type: Classified task type
            entities: Extracted entities
        
        Returns:
            Workflow name
        """
        workflows = {
            TaskType.QUESTION_ANSWER: 'simple_qa',
            TaskType.QUIZ_GENERATION: 'generate_quiz' if 'email' not in entities else 'generate_and_email_quiz',
            TaskType.CONCEPT_EXPLANATION: 'explain_concept' if 'format' not in entities else 'explain_and_generate_doc',
            TaskType.DOCUMENT_GENERATION: 'generate_document',
            TaskType.EMAIL_SEND: 'send_email',
            TaskType.WORKFLOW_COMPLEX: 'custom_workflow',
            TaskType.GREETING: 'greeting',
            TaskType.HELP: 'help',
        }
        
        return workflows.get(task_type, 'unknown')
    
    def _build_parameters(
        self, 
        task_type: TaskType, 
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build parameters for task execution.
        
        Args:
            task_type: Classified task type
            entities: Extracted entities
            context: Conversation context
        
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        # Add extracted entities
        params.update(entities)
        
        # Add defaults based on task type
        if task_type == TaskType.QUIZ_GENERATION:
            params.setdefault('num_questions', 5)
            params.setdefault('format', 'pdf')
            params.setdefault('grade_level', context.get('grade_level') if context else None)
        
        elif task_type == TaskType.CONCEPT_EXPLANATION:
            params.setdefault('explanation_type', 'step-by-step')
            params.setdefault('include_examples', True)
        
        elif task_type == TaskType.QUESTION_ANSWER:
            params.setdefault('top_k', 5)
        
        # Add context if available
        if context:
            params['context'] = context
        
        return params
    
    def _generate_clarification_options(self, query: str) -> List[str]:
        """Generate clarification options for ambiguous queries."""
        return [
            "Would you like me to answer a question about this topic?",
            "Would you like me to generate a quiz on this topic?",
            "Would you like a detailed explanation of this concept?",
            "Would you like me to create study materials?"
        ]
    
    def get_task_description(self, task_type: str) -> str:
        """Get human-readable description of task type."""
        descriptions = {
            'question_answer': 'Answer a geometry question',
            'quiz_generation': 'Generate a geometry quiz',
            'concept_explanation': 'Explain a geometry concept',
            'document_generation': 'Create a document (PDF/DOCX/PPT)',
            'email_send': 'Send content via email',
            'workflow_complex': 'Execute a multi-step workflow',
            'greeting': 'Respond to greeting',
            'help': 'Provide system help',
            'unknown': 'Unknown task type'
        }
        return descriptions.get(task_type, 'Unknown task')


class ConversationMemory:
    """
    Manages conversation history and context across multiple turns.
    Tracks entities, references, and user preferences.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to remember
        """
        self.max_history = max_history
        self.history = []
        self.entities = {}  # Persistent entities (grade, topic, etc.)
        self.user_preferences = {}
        
        logger.info("Conversation Memory initialized")
    
    def add_turn(
        self, 
        user_message: str, 
        assistant_response: str,
        task_type: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ):
        """
        Add a conversation turn to memory.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            task_type: Type of task executed
            entities: Entities extracted from this turn
        """
        turn = {
            'user': user_message,
            'assistant': assistant_response,
            'task_type': task_type,
            'entities': entities or {},
            'timestamp': self._get_timestamp()
        }
        
        self.history.append(turn)
        
        # Update persistent entities
        if entities:
            self.entities.update(entities)
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, num_turns: int = 3) -> Dict[str, Any]:
        """
        Get conversation context for current turn.
        
        Args:
            num_turns: Number of recent turns to include
        
        Returns:
            Dictionary with context information
        """
        recent_history = self.history[-num_turns:] if num_turns > 0 else []
        
        return {
            'recent_history': recent_history,
            'entities': self.entities.copy(),
            'user_preferences': self.user_preferences.copy(),
            'conversation_length': len(self.history)
        }
    
    def get_entity(self, entity_name: str, default: Any = None) -> Any:
        """Get a specific entity from memory."""
        return self.entities.get(entity_name, default)
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.user_preferences[key] = value
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.entities = {}
        self.user_preferences = {}
        logger.info("Conversation memory cleared")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def resolve_references(self, query: str) -> str:
        """
        Resolve references in query (e.g., "it", "that", "this topic").
        
        Args:
            query: User's query with possible references
        
        Returns:
            Query with resolved references
        """
        # Simple reference resolution
        if not self.history:
            return query
        
        query_lower = query.lower()
        
        # Resolve "it" / "that" with last mentioned topic
        if any(word in query_lower for word in ['it', 'that', 'this']):
            last_topic = self.entities.get('topic')
            if last_topic:
                query = re.sub(r'\b(it|that|this)\b', last_topic, query, flags=re.IGNORECASE, count=1)
        
        # Resolve "the same" with last parameters
        if 'same' in query_lower:
            if 'grade_level' in self.entities:
                query += f" for {self.entities['grade_level']}"
        
        return query
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.history:
            return "No conversation history."
        
        summary = f"Conversation has {len(self.history)} turns.\n"
        
        if self.entities:
            summary += f"Known entities: {', '.join(f'{k}={v}' for k, v in self.entities.items())}\n"
        
        task_counts = {}
        for turn in self.history:
            task = turn.get('task_type', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        summary += f"Tasks executed: {task_counts}"
        
        return summary