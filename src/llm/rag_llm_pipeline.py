#!/usr/bin/env python
"""
Merged RAG-LLM Pipeline - Complete Geometry Tutor
Combines retrieval with LLM generation for comprehensive Q&A system
"""

import os
import logging
from typing import Dict, Any, Optional, List
import time

from ..retrieval.rag_pipeline import GeometryRAGPipeline, RetrievalConfig, RetrievalStrategy
from .gemini_client import GeminiClient
from ..config.settings import settings

# Setup logging
log_file = getattr(
    settings, 
    'log_file', 
    os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'rag_llm_pipeline.log')
)
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


class GeometryTutorPipeline:
    """
    Complete pipeline integrating RAG retrieval with LLM generation.
    Handles Q&A, quiz generation, explanations, and conversations.
    
    Features:
    - Multi-step reasoning
    - Context-aware responses
    - Adaptive difficulty
    - Conversation memory
    - Error recovery
    """
    
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None,
        enable_reranker: bool = True,
        default_top_k: int = 5
    ):
        """
        Initialize the complete tutor pipeline.
        
        Args:
            gemini_api_key: Google Gemini API key
            enable_reranker: Whether to use reranking for better results
            default_top_k: Default number of chunks to retrieve
        """
        logger.info("="*70)
        logger.info("Initializing Geometry Tutor Pipeline")
        logger.info("="*70)
        
        # Initialize RAG pipeline
        try:
            self.rag_pipeline = GeometryRAGPipeline(enable_reranker=enable_reranker)
            logger.info("✓ RAG pipeline initialized")
            print("✓ RAG pipeline initialized")
        except Exception as e:
            logger.error(f"✗ RAG pipeline initialization failed: {e}")
            raise
        
        # Initialize LLM client
        try:
            gemini_api_key = gemini_api_key or getattr(settings, 'GEMINI_API_KEY', None)
            self.llm_client = GeminiClient(api_key=gemini_api_key)
            logger.info("✓ Gemini LLM initialized")
            print("✓ Gemini LLM initialized")
        except Exception as e:
            logger.error(f"✗ Gemini initialization failed: {e}")
            raise
        
        # Configuration
        self.default_top_k = default_top_k
        self.enable_reranker = enable_reranker
        
        # Test system components
        self._test_system()
        
        logger.info("="*70)
        logger.info("✓ Geometry Tutor Pipeline ready!")
        logger.info("="*70)
        print("\n✓ Geometry Tutor Pipeline ready!\n")
    
    # ========== Core Methods ==========
    
    def answer_question(
        self,
        query: str,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> Dict[str, Any]:
        """
        Answer a geometry question using RAG + LLM.
        
        Args:
            query: Student's question
            grade_level: Target grade level (e.g., "Grade 7")
            difficulty: Content difficulty (Beginner/Intermediate/Advanced)
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source references
            retrieval_strategy: Retrieval strategy to use
        
        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()
        logger.info(f"Answering question: {query}")
        
        if top_k is None:
            top_k = self.default_top_k
        
        # Step 1: Retrieve relevant context
        retrieval_config = RetrievalConfig(
            strategy=retrieval_strategy,
            top_k=top_k * 2,  # Retrieve more, then rerank
            rerank=self.enable_reranker,
            rerank_top_k=top_k,
            filters=self._build_filters(grade_level, difficulty)
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(query, retrieval_config)
            logger.info(f"Retrieved {len(retrieval_result.chunks)} relevant chunks")
            
            if not retrieval_result.chunks:
                logger.warning("No relevant context found")
                return {
                    'query': query,
                    'answer': "I couldn't find relevant information in my knowledge base. Could you rephrase your question or ask about a different geometry topic?",
                    'sources': [],
                    'retrieval_metadata': {
                        'num_chunks': 0,
                        'sources': [],
                        'avg_score': 0
                    },
                    'success': False,
                    'error': 'No relevant context found'
                }
            
            # Step 2: Generate answer using LLM
            llm_response = self.llm_client.generate_answer(
                query=query,
                context=retrieval_result.context,
                grade_level=grade_level or self._infer_grade_level(retrieval_result),
                difficulty=difficulty or self._infer_difficulty(retrieval_result)
            )
            
            elapsed_time = time.time() - start_time
            
            # Step 3: Compile comprehensive response
            response = {
                'query': query,
                'answer': llm_response.get('answer', 'Error generating answer'),
                'context_preview': self._truncate_context(retrieval_result.context, 500),
                'retrieval_metadata': {
                    'num_chunks': len(retrieval_result.chunks),
                    'sources': retrieval_result.metadata.get('sources', []),
                    'grade_levels': retrieval_result.metadata.get('grade_levels', []),
                    'topics': retrieval_result.metadata.get('topics', []),
                    'avg_score': retrieval_result.metadata.get('avg_score', 0),
                    'strategy': retrieval_strategy.value
                },
                'generation_metadata': {
                    'model': llm_response.get('model'),
                    'generation_time': llm_response.get('generation_time', 0),
                },
                'total_time': elapsed_time,
                'success': llm_response.get('success', False)
            }
            
            # Add detailed source references if requested
            if include_sources:
                response['sources'] = self._format_sources(retrieval_result.chunks[:3])
            
            logger.info(f"✓ Answer generated successfully in {elapsed_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                'query': query,
                'answer': f"An error occurred while processing your question: {str(e)}",
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def generate_quiz(
        self,
        topic: str,
        grade_level: str,
        num_questions: int = 5,
        difficulty: Optional[str] = None,
        question_types: Optional[List[str]] = None,
        use_structured_output: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a quiz on a specific geometry topic.
        
        Args:
            topic: Topic for quiz (e.g., "Triangles", "Circles")
            grade_level: Target grade level
            num_questions: Number of questions (default 5)
            difficulty: Optional difficulty filter
            question_types: Types of questions to include
            use_structured_output: Return structured JSON format
        
        Returns:
            Dict with quiz content and metadata
        """
        start_time = time.time()
        logger.info(f"Generating quiz on: {topic} for {grade_level}")
        
        # Step 1: Retrieve comprehensive context about the topic
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.KEYWORD_ONLY,
            top_k=15,  # More context for quiz generation
            rerank=self.enable_reranker,
            rerank_top_k=10,
            filters=self._build_filters(grade_level, difficulty)
        )
        
        try:
            # Use a detailed query to get comprehensive coverage
            # search_query = f"Explain {topic} concepts, formulas, theorems, and properties"
            search_query = topic
            retrieval_result = self.rag_pipeline.retrieve(search_query, retrieval_config)

            print(f"Search query for debugging: {search_query}")
            print(f"Retrieved {len(retrieval_result.chunks)} chunks for quiz generation")

            if not retrieval_result.chunks:
                logger.warning(f"No content found for topic: {topic}")
                return {
                    'quiz': None,
                    'topic': topic,
                    'success': False,
                    'error': f'No content found for topic: {topic}'
                }
            
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for quiz generation")
            print(f"Retrieved context for debugging: {retrieval_result.context}")
            # Step 2: Generate quiz using LLM
            quiz_response = self.llm_client.generate_quiz(
                topic=topic,
                context=retrieval_result.context,
                grade_level=grade_level,
                num_questions=num_questions,
                question_types=question_types,
                use_structured_output=use_structured_output
            )
            
            elapsed_time = time.time() - start_time
            
            # Step 3: Add metadata and compile response
            if quiz_response.get('success'):
                quiz_response['metadata'] = {
                    **quiz_response.get('metadata', {}),
                    'sources': retrieval_result.metadata.get('sources', []),
                    'num_chunks_used': len(retrieval_result.chunks),
                    'retrieval_score': retrieval_result.metadata.get('avg_score', 0),
                    'generation_time': elapsed_time
                }
                logger.info(f"✓ Quiz generated successfully in {elapsed_time:.2f}s")
            else:
                logger.error(f"Quiz generation failed: {quiz_response.get('error')}")
            
            return quiz_response
            
        except Exception as e:
            logger.error(f"Error in generate_quiz: {e}", exc_info=True)
            return {
                'quiz': None,
                'topic': topic,
                'success': False,
                'error': str(e)
            }
    
    def explain_concept(
        self,
        concept: str,
        grade_level: str,
        explanation_type: str = "step-by-step",
        include_examples: bool = True,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a geometry concept.
        
        Args:
            concept: Concept to explain (e.g., "Pythagorean Theorem")
            grade_level: Target grade level
            explanation_type: Type of explanation 
                ("step-by-step", "visual", "proof", "example")
            include_examples: Whether to include worked examples
            difficulty: Optional difficulty level
        
        Returns:
            Dict with explanation and metadata
        """
        start_time = time.time()
        logger.info(f"Explaining: {concept} ({explanation_type})")
        
        # Step 1: Retrieve context with hierarchical strategy for concept explanations
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.HIERARCHICAL,
            top_k=12,
            rerank=self.enable_reranker,
            rerank_top_k=8,
            include_parents=True,  # Include parent chunks for context
            filters=self._build_filters(grade_level, difficulty)
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(concept, retrieval_config)
            
            if not retrieval_result.chunks:
                logger.warning(f"No information found for concept: {concept}")
                return {
                    'concept': concept,
                    'explanation': f"I don't have detailed information about '{concept}' in my knowledge base for {grade_level}.",
                    'success': False,
                    'error': 'No relevant content found'
                }
            
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for explanation")
            
            # Step 2: Generate explanation
            explanation_response = self.llm_client.generate_explanation(
                concept=concept,
                context=retrieval_result.context,
                grade_level=grade_level,
                explanation_type=explanation_type,
                include_examples=include_examples
            )
            
            elapsed_time = time.time() - start_time
            
            # Step 3: Compile response with metadata
            response = {
                'concept': concept,
                'explanation': explanation_response.get('explanation'),
                'grade_level': grade_level,
                'type': explanation_type,
                'metadata': {
                    'sources': retrieval_result.metadata.get('sources', []),
                    'topics': retrieval_result.metadata.get('topics', []),
                    'num_chunks_used': len(retrieval_result.chunks),
                    'retrieval_score': retrieval_result.metadata.get('avg_score', 0),
                    'model': explanation_response.get('model'),
                    'generation_time': elapsed_time
                },
                'success': explanation_response.get('success', False)
            }
            
            if response['success']:
                logger.info(f"✓ Explanation generated successfully in {elapsed_time:.2f}s")
            else:
                logger.error(f"Explanation generation failed: {explanation_response.get('error')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in explain_concept: {e}", exc_info=True)
            return {
                'concept': concept,
                'explanation': None,
                'success': False,
                'error': str(e)
            }
    
    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieve_context: bool = True,
        grade_level: Optional[str] = None,
        context_window: int = 5
    ) -> Dict[str, Any]:
        """
        Handle conversational interaction with memory and context.
        
        Args:
            message: User's message
            chat_history: Previous conversation history
            retrieve_context: Whether to retrieve relevant context from RAG
            grade_level: Optional grade filter
            context_window: Number of recent turns to include (default 5)
        
        Returns:
            Dict with response and updated history
        """
        start_time = time.time()
        logger.info(f"Chat message: {message}")
        
        context = None
        retrieval_metadata = None
        
        # Step 1: Retrieve context if requested
        if retrieve_context:
            try:
                retrieval_config = RetrievalConfig(
                    strategy=RetrievalStrategy.HYBRID,
                    top_k=5,
                    rerank=self.enable_reranker,
                    rerank_top_k=3,
                    filters={'grade_level': grade_level} if grade_level else None
                )
                
                retrieval_result = self.rag_pipeline.retrieve(message, retrieval_config)
                
                if retrieval_result.chunks:
                    # Limit context to prevent overwhelming the prompt
                    context = self._truncate_context(retrieval_result.context, 1000)
                    retrieval_metadata = {
                        'num_chunks': len(retrieval_result.chunks),
                        'sources': retrieval_result.metadata.get('sources', [])[:2],
                        'avg_score': retrieval_result.metadata.get('avg_score', 0)
                    }
                    logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for chat context")
                    
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
                # Continue without context
        
        # Step 2: Generate chat response
        try:
            chat_response = self.llm_client.chat(
                message=message,
                chat_history=chat_history or [],
                context=context,
                grade_level=grade_level
            )
            
            elapsed_time = time.time() - start_time
            
            # Add metadata
            chat_response['metadata'] = {
                'retrieval_used': retrieve_context and context is not None,
                'retrieval_metadata': retrieval_metadata,
                'response_time': elapsed_time
            }
            
            logger.info(f"✓ Chat response generated in {elapsed_time:.2f}s")
            return chat_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return {
                'response': "I'm sorry, I encountered an error. Could you rephrase that?",
                'chat_history': chat_history or [],
                'success': False,
                'error': str(e)
            }
    
    # ========== Batch & Advanced Methods ==========
    
    def batch_answer_questions(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of question dicts with 'query' and optional 'grade_level'
            show_progress: Whether to print progress
        
        Returns:
            List of answer dicts
        """
        logger.info(f"Batch processing {len(questions)} questions")
        results = []
        
        for i, q in enumerate(questions, 1):
            if show_progress:
                print(f"Processing question {i}/{len(questions)}: {q.get('query', '')[:50]}...")
            
            result = self.answer_question(
                query=q.get('query'),
                grade_level=q.get('grade_level'),
                difficulty=q.get('difficulty')
            )
            results.append(result)
        
        logger.info(f"✓ Batch processing complete: {len(results)} answers generated")
        return results
    
    def adaptive_explanation(
        self,
        concept: str,
        student_level: str,
        previous_attempts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate adaptive explanation based on student level and previous attempts.
        Uses different explanation types based on student understanding.
        
        Args:
            concept: Concept to explain
            student_level: Student's current level (Beginner/Intermediate/Advanced)
            previous_attempts: Previous explanation types tried
        
        Returns:
            Explanation with adaptive approach
        """
        if previous_attempts is None:
            previous_attempts = []
        
        # Choose explanation type based on level and attempts
        explanation_types = {
            'Beginner': ['step-by-step', 'visual', 'example'],
            'Intermediate': ['step-by-step', 'example', 'visual'],
            'Advanced': ['proof', 'example', 'step-by-step']
        }
        
        available_types = [
            t for t in explanation_types.get(student_level, ['step-by-step'])
            if t not in previous_attempts
        ]
        
        if not available_types:
            available_types = ['step-by-step']  # Fallback
        
        explanation_type = available_types[0]
        
        logger.info(f"Adaptive explanation: {concept} using {explanation_type} for {student_level}")
        
        return self.explain_concept(
            concept=concept,
            grade_level=self._level_to_grade(student_level),
            explanation_type=explanation_type,
            include_examples=True
        )
    
    # ========== Utility Methods ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics and health check."""
        try:
            rag_stats = self.rag_pipeline.get_statistics()
            
            return {
                'status': 'healthy',
                'rag_pipeline': rag_stats,
                'llm_client': {
                    'model': self.llm_client.model_name,
                    'api_configured': bool(self.llm_client.api_key)
                },
                'configuration': {
                    'default_top_k': self.default_top_k,
                    'reranker_enabled': self.enable_reranker
                }
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _build_filters(
        self, 
        grade_level: Optional[str], 
        difficulty: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build filter dictionary for retrieval."""
        filters = {}
        
        if grade_level:
            filters['grade_level'] = grade_level
        
        if difficulty:
            filters['difficulty'] = difficulty
        
        return filters if filters else None
    
    def _infer_grade_level(self, retrieval_result) -> Optional[str]:
        """Infer grade level from retrieval results."""
        grade_levels = retrieval_result.metadata.get('grade_levels', [])
        return grade_levels[0] if grade_levels else None
    
    def _infer_difficulty(self, retrieval_result) -> Optional[str]:
        """Infer difficulty from retrieval results."""
        difficulties = retrieval_result.metadata.get('difficulties', [])
        return difficulties[0] if difficulties else None
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context to maximum length."""
        if len(context) <= max_length:
            return context
        return context[:max_length] + "..."
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Format source references from chunks."""
        sources = []
        for chunk in chunks:
            content = chunk.get('content', {})
            sources.append({
                'source': content.get('source', 'Unknown'),
                'grade_level': content.get('grade_level', 'Unknown'),
                'topic': content.get('topic', 'General'),
                'score': chunk.get('score', 0)
            })
        return sources
    
    def _level_to_grade(self, level: str) -> str:
        """Convert difficulty level to grade level."""
        mapping = {
            'Beginner': 'Grade 6',
            'Intermediate': 'Grade 8',
            'Advanced': 'Grade 10'
        }
        return mapping.get(level, 'Grade 8')
    
    def _test_system(self):
        """Test that all components are working."""
        logger.info("\nTesting system components...")
        
        # Test RAG
        try:
            test_result = self.rag_pipeline.get_statistics()
            total_chunks = test_result.get('total_chunks', 0)
            logger.info(f"✓ RAG system ready: {total_chunks} chunks indexed")
            print(f"✓ RAG system ready: {total_chunks} chunks indexed")
        except Exception as e:
            logger.error(f"✗ RAG system test failed: {e}")
            print(f"✗ RAG system test failed: {e}")
        
        # Test LLM
        try:
            if self.llm_client.test_connection():
                logger.info("✓ LLM system ready")
            else:
                logger.warning("⚠ LLM test produced warnings")
        except Exception as e:
            logger.error(f"✗ LLM test failed: {e}")
            print(f"✗ LLM test failed: {e}")
        
        logger.info("System test complete\n")