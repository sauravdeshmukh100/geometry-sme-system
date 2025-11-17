#!/usr/bin/env python
"""
Enhanced RAG-LLM Pipeline - Complete Geometry Tutor
Compatible with improved hierarchical retrieval, adaptive strategies, and deduplication
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
    
    Enhanced Features:
    - Adaptive retrieval strategies
    - Level-aware fusion for hierarchical retrieval
    - Automatic deduplication
    - Token-aware context assembly
    - Multi-step reasoning
    - Conversation memory
    - Error recovery
    """
    
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None,
        enable_reranker: bool = True,
        default_top_k: int = 5,
        max_context_tokens: int = 4000
    ):
        """
        Initialize the complete tutor pipeline.
        
        Args:
            gemini_api_key: Google Gemini API key
            enable_reranker: Whether to use reranking for better results
            default_top_k: Default number of chunks to retrieve
            max_context_tokens: Maximum tokens for context assembly
        """
        logger.info("="*70)
        logger.info("Initializing Enhanced Geometry Tutor Pipeline")
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
        self.max_context_tokens = max_context_tokens
        self.embedding_limit = getattr(settings, 'EMBEDDING_MAX_TOKENS', 384)
        
        # Test system components
        self._test_system()
        
        logger.info("="*70)
        logger.info("✓ Enhanced Geometry Tutor Pipeline ready!")
        logger.info(f"  Embedding limit: {self.embedding_limit} tokens")
        logger.info(f"  Max context: {self.max_context_tokens} tokens")
        logger.info("="*70)
        print(f"\n✓ Enhanced Geometry Tutor Pipeline ready!")
        print(f"  Enhanced features: Adaptive strategies, Level fusion, Deduplication\n")
    
    # ========== Core Methods ==========
    
    def answer_question(
        self,
        query: str,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
        enable_deduplication: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a geometry question using RAG + LLM with enhanced retrieval.
        
        Args:
            query: Student's question
            grade_level: Target grade level (e.g., "Grade 7")
            difficulty: Content difficulty (Beginner/Intermediate/Advanced)
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source references
            retrieval_strategy: Retrieval strategy (ADAPTIVE recommended)
            enable_deduplication: Remove duplicate/similar chunks
        
        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()
        logger.info(f"Answering question: {query}")
        
        if top_k is None:
            top_k = self.default_top_k
        
        # Step 1: Retrieve relevant context with enhanced config
        retrieval_config = RetrievalConfig(
            strategy=retrieval_strategy,
            top_k=top_k * 2,  # Retrieve more, then rerank
            rerank=self.enable_reranker,
            rerank_top_k=top_k,
            use_metadata_boost=True,
            level_preference=None,  # Let adaptive choose
            filters=self._build_filters(grade_level, difficulty),
            embeddable_only=True,
            max_context_tokens=self.max_context_tokens,
            enable_deduplication=enable_deduplication,
            similarity_threshold=0.95
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
                        'avg_score': 0,
                        'strategy_used': retrieval_strategy.value
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
            
            # Step 3: Compile comprehensive response with enhanced metadata
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
                    'strategy_used': retrieval_result.metadata.get('strategy_used', retrieval_strategy.value),
                    'level_preference': retrieval_result.metadata.get('level_preference'),
                    'embeddable_count': retrieval_result.metadata.get('embeddable_count', 0),
                    'context_only_count': retrieval_result.metadata.get('context_only_count', 0),
                    'level_distribution': retrieval_result.metadata.get('level_distribution', {}),
                    'deduplication_enabled': enable_deduplication
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
        use_structured_output: bool = True,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> Dict[str, Any]:
        """
        Generate a quiz on a specific geometry topic with enhanced retrieval.
        
        Args:
            topic: Topic for quiz (e.g., "Triangles", "Circles")
            grade_level: Target grade level
            num_questions: Number of questions (default 5)
            difficulty: Optional difficulty filter
            question_types: Types of questions to include
            use_structured_output: Return structured JSON format
            retrieval_strategy: Strategy for content retrieval
        
        Returns:
            Dict with quiz content and metadata
        """
        start_time = time.time()
        logger.info(f"Generating quiz on: {topic} for {grade_level}")
        
        # Step 1: Retrieve comprehensive context about the topic
        retrieval_config = RetrievalConfig(
            strategy=retrieval_strategy,
            top_k=15,  # More context for quiz generation
            rerank=self.enable_reranker,
            rerank_top_k=10,
            use_metadata_boost=True,
            include_parents=False,
            filters=self._build_filters(grade_level, difficulty),
            embeddable_only=True,
            max_context_tokens=self.max_context_tokens,
            enable_deduplication=True,
            similarity_threshold=0.90  # Allow slightly more variety for quiz
        )
        
        try:
            # Use the topic directly as search query
            search_query = topic
            retrieval_result = self.rag_pipeline.retrieve(search_query, retrieval_config)

            logger.debug(f"Search query: {search_query}")
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for quiz generation")

            if not retrieval_result.chunks:
                logger.warning(f"No content found for topic: {topic}")
                return {
                    'quiz': None,
                    'topic': topic,
                    'success': False,
                    'error': f'No content found for topic: {topic}',
                    'metadata': {
                        'retrieval_strategy': retrieval_strategy.value,
                        'num_chunks': 0
                    }
                }
            
            # Log context preview for debugging
            logger.debug(f"Context preview: {retrieval_result.context[:200]}...")
            
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
            
            # Step 3: Add enhanced metadata and compile response
            if quiz_response.get('success'):
                quiz_response['metadata'] = {
                    **quiz_response.get('metadata', {}),
                    'sources': retrieval_result.metadata.get('sources', []),
                    'num_chunks_used': len(retrieval_result.chunks),
                    'retrieval_score': retrieval_result.metadata.get('avg_score', 0),
                    'retrieval_strategy': retrieval_result.metadata.get('strategy_used', retrieval_strategy.value),
                    'level_distribution': retrieval_result.metadata.get('level_distribution', {}),
                    'embeddable_count': retrieval_result.metadata.get('embeddable_count', 0),
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
                'error': str(e),
                'metadata': {
                    'retrieval_strategy': retrieval_strategy.value
                }
            }
    
    def explain_concept(
        self,
        concept: str,
        grade_level: str,
        explanation_type: str = "step-by-step",
        include_examples: bool = True,
        difficulty: Optional[str] = None,
        use_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a geometry concept with enhanced retrieval.
        
        Args:
            concept: Concept to explain (e.g., "Pythagorean Theorem")
            grade_level: Target grade level
            explanation_type: Type of explanation 
                ("step-by-step", "visual", "proof", "example")
            include_examples: Whether to include worked examples
            difficulty: Optional difficulty level
            use_hierarchical: Use hierarchical retrieval with level fusion
        
        Returns:
            Dict with explanation and metadata
        """
        start_time = time.time()
        logger.info(f"Explaining: {concept} ({explanation_type})")
        
        # Step 1: Retrieve context with hierarchical strategy for concept explanations
        retrieval_strategy = RetrievalStrategy.HIERARCHICAL if use_hierarchical else RetrievalStrategy.ADAPTIVE
        
        retrieval_config = RetrievalConfig(
            strategy=retrieval_strategy,
            top_k=12,
            rerank=self.enable_reranker,
            rerank_top_k=8,
            use_metadata_boost=True,
            include_parents=True,  # Include parent chunks for broader context
            include_children=False,
            level_preference=None,  # Let hierarchical fusion work
            filters=self._build_filters(grade_level, difficulty),
            embeddable_only=True,
            max_context_tokens=self.max_context_tokens,
            enable_deduplication=True,
            similarity_threshold=0.92
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(concept, retrieval_config)

            for i, chunk in enumerate(retrieval_result.chunks):
                print(f"Chunk {i+1}: Score={chunk.get('score')}, Level={chunk.get('content', {}).get('level')}, Source={chunk.get('content', {}).get('source')}, Length={len(chunk.get('content', {}).get('text', ''))}")
            
            if not retrieval_result.chunks:
                logger.warning(f"No information found for concept: {concept}")
                return {
                    'concept': concept,
                    'explanation': f"I don't have detailed information about '{concept}' in my knowledge base for {grade_level}.",
                    'success': False,
                    'error': 'No relevant content found',
                    'metadata': {
                        'retrieval_strategy': retrieval_strategy.value
                    }
                }
            
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for explanation bro")
            print("Context before passing to LLM:")
            print(retrieval_result.context)
            
            # Step 2: Generate explanation
            explanation_response = self.llm_client.generate_explanation(
                concept=concept,
                context=retrieval_result.context,
                grade_level=grade_level,
                explanation_type=explanation_type,
                include_examples=include_examples
            )
            
            elapsed_time = time.time() - start_time
            
            # Step 3: Compile response with enhanced metadata
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
                    'retrieval_strategy': retrieval_result.metadata.get('strategy_used', retrieval_strategy.value),
                    'level_distribution': retrieval_result.metadata.get('level_distribution', {}),
                    'embeddable_count': retrieval_result.metadata.get('embeddable_count', 0),
                    'context_only_count': retrieval_result.metadata.get('context_only_count', 0),
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
                'error': str(e),
                'metadata': {
                    'retrieval_strategy': retrieval_strategy.value
                }
            }
    
    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieve_context: bool = True,
        grade_level: Optional[str] = None,
        context_window: int = 5,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    ) -> Dict[str, Any]:
        """
        Handle conversational interaction with memory and enhanced context retrieval.
        
        Args:
            message: User's message
            chat_history: Previous conversation history
            retrieve_context: Whether to retrieve relevant context from RAG
            grade_level: Optional grade filter
            context_window: Number of recent turns to include (default 5)
            retrieval_strategy: Strategy for context retrieval
        
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
                    strategy=retrieval_strategy,
                    top_k=5,
                    rerank=self.enable_reranker,
                    rerank_top_k=3,
                    use_metadata_boost=True,
                    filters={'grade_level': grade_level} if grade_level else None,
                    embeddable_only=True,
                    max_context_tokens=2000,  # Limit for chat to leave room for history
                    enable_deduplication=True,
                    similarity_threshold=0.95
                )
                
                retrieval_result = self.rag_pipeline.retrieve(message, retrieval_config)
                
                if retrieval_result.chunks:
                    # Limit context to prevent overwhelming the prompt
                    context = retrieval_result.context
                    retrieval_metadata = {
                        'num_chunks': len(retrieval_result.chunks),
                        'sources': retrieval_result.metadata.get('sources', [])[:2],
                        'avg_score': retrieval_result.metadata.get('avg_score', 0),
                        'strategy_used': retrieval_result.metadata.get('strategy_used', retrieval_strategy.value),
                        'level_distribution': retrieval_result.metadata.get('level_distribution', {})
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
        Answer multiple questions in batch with enhanced retrieval.
        
        Args:
            questions: List of question dicts with 'query' and optional parameters
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
                difficulty=q.get('difficulty'),
                retrieval_strategy=q.get('retrieval_strategy', RetrievalStrategy.ADAPTIVE)
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
        Uses different explanation types and retrieval strategies based on understanding.
        
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
        
        # Use hierarchical for complex concepts, adaptive for simpler ones
        use_hierarchical = explanation_type in ['proof', 'step-by-step']
        
        logger.info(f"Adaptive explanation: {concept} using {explanation_type} for {student_level}")
        
        return self.explain_concept(
            concept=concept,
            grade_level=self._level_to_grade(student_level),
            explanation_type=explanation_type,
            include_examples=True,
            use_hierarchical=use_hierarchical
        )
    
    def compare_retrieval_strategies(
        self,
        query: str,
        grade_level: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare different retrieval strategies for a query.
        Useful for debugging and optimization.
        
        Args:
            query: Query to test
            grade_level: Optional grade filter
            top_k: Number of results per strategy
        
        Returns:
            Comparison results for each strategy
        """
        logger.info(f"Comparing strategies for: {query}")
        
        strategies = [
            RetrievalStrategy.ADAPTIVE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.VECTOR_ONLY
        ]
        
        results = {}
        
        for strategy in strategies:
            config = RetrievalConfig(
                strategy=strategy,
                top_k=top_k * 2,
                rerank=self.enable_reranker,
                rerank_top_k=top_k,
                filters={'grade_level': grade_level} if grade_level else None,
                max_context_tokens=self.max_context_tokens,
                enable_deduplication=True
            )
            
            try:
                start = time.time()
                result = self.rag_pipeline.retrieve(query, config)
                elapsed = time.time() - start
                
                results[strategy.value] = {
                    'num_chunks': len(result.chunks),
                    'avg_score': result.metadata.get('avg_score', 0),
                    'sources': result.metadata.get('sources', []),
                    'level_distribution': result.metadata.get('level_distribution', {}),
                    'retrieval_time': elapsed,
                    'success': True
                }
                
                logger.info(f"  {strategy.value}: {len(result.chunks)} chunks, avg_score={result.metadata.get('avg_score', 0):.4f}")
                
            except Exception as e:
                logger.error(f"  {strategy.value}: Failed - {e}")
                results[strategy.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'query': query,
            'grade_level': grade_level,
            'top_k': top_k,
            'strategies': results,
            'recommendation': self._recommend_strategy(results)
        }
    
    def _recommend_strategy(self, results: Dict[str, Any]) -> str:
        """Recommend best strategy based on comparison results."""
        best_strategy = None
        best_score = 0
        
        for strategy, data in results.items():
            if data.get('success') and data.get('avg_score', 0) > best_score:
                best_score = data['avg_score']
                best_strategy = strategy
        
        return best_strategy or 'adaptive'
    
    # ========== Utility Methods ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics and health check."""
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
                    'max_context_tokens': self.max_context_tokens,
                    'embedding_limit': self.embedding_limit,
                    'reranker_enabled': self.enable_reranker
                },
                'enhanced_features': {
                    'adaptive_strategy': True,
                    'hierarchical_retrieval': True,
                    'level_aware_fusion': True,
                    'deduplication': True,
                    'token_aware_assembly': True
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
        """Format source references from chunks with enhanced metadata."""
        sources = []
        for chunk in chunks:
            content = chunk.get('content', {})
            sources.append({
                'source': content.get('source', 'Unknown'),
                'grade_level': content.get('grade_level', 'Unknown'),
                'topic': content.get('topic', 'General'),
                'difficulty': content.get('difficulty', 'Unknown'),
                'level': content.get('level', '?'),
                'embeddable': content.get('embeddable', True),
                'score': chunk.get('score', 0),
                'rerank_score': chunk.get('rerank_score')
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
        """Test that all components are working with enhanced features."""
        logger.info("\nTesting enhanced system components...")
        
        # Test RAG
        try:
            test_result = self.rag_pipeline.get_statistics()
            total_chunks = test_result.get('total_chunks', 0)
            embeddable_chunks = test_result.get('embeddable_chunks', 0)
            context_chunks = test_result.get('context_chunks', 0)
            
            logger.info(f"✓ RAG system ready:")
            logger.info(f"  Total chunks: {total_chunks}")
            logger.info(f"  Embeddable: {embeddable_chunks}")
            logger.info(f"  Context-only: {context_chunks}")
            
            print(f"✓ RAG system ready: {total_chunks} chunks indexed")
            print(f"  Embeddable: {embeddable_chunks}, Context: {context_chunks}")
            
            # Check enhanced features
            features = test_result.get('features', {})
            if features:
                logger.info("  Enhanced features:")
                for feature, enabled in features.items():
                    if enabled:
                        logger.info(f"    ✓ {feature.replace('_', ' ').title()}")
                        
        except Exception as e:
            logger.error(f"✗ RAG system test failed: {e}")
            print(f"✗ RAG system test failed: {e}")
        
        # Test LLM
        try:
            if self.llm_client.test_connection():
                logger.info("✓ LLM system ready")
                print(f"✓ LLM system ready: {self.llm_client.model_name}")
            else:
                logger.warning("⚠ LLM test produced warnings")
                print("⚠ LLM test produced warnings")
        except Exception as e:
            logger.error(f"✗ LLM test failed: {e}")
            print(f"✗ LLM test failed: {e}")
        
        logger.info("System test complete\n")


# ========== Convenience Functions ==========

def create_geometry_tutor(
    gemini_api_key: Optional[str] = None,
    enable_reranker: bool = True,
    **kwargs
) -> GeometryTutorPipeline:
    """
    Convenience function to create a GeometryTutorPipeline instance.
    
    Args:
        gemini_api_key: Google Gemini API key
        enable_reranker: Whether to enable reranking
        **kwargs: Additional configuration options
    
    Returns:
        Initialized GeometryTutorPipeline
    """
    return GeometryTutorPipeline(
        gemini_api_key=gemini_api_key,
        enable_reranker=enable_reranker,
        **kwargs
    )


# ========== Example Usage ==========

if __name__ == "__main__":
    """
    Example usage of the enhanced RAG-LLM pipeline.
    """
    print("="*70)
    print("GEOMETRY TUTOR PIPELINE - ENHANCED DEMO")
    print("="*70)
    
    try:
        # Initialize pipeline
        tutor = GeometryTutorPipeline(enable_reranker=True)
        
        # Example 1: Answer a question with adaptive strategy
        print("\n" + "="*70)
        print("EXAMPLE 1: Question Answering (Adaptive Strategy)")
        print("="*70)
        
        result = tutor.answer_question(
            query="What is the Pythagorean theorem and how do I use it?",
            grade_level="Grade 8",
            retrieval_strategy=RetrievalStrategy.ADAPTIVE
        )
        
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer'][:300]}...")
        print(f"\nMetadata:")
        print(f"  Strategy Used: {result['retrieval_metadata']['strategy_used']}")
        print(f"  Chunks Retrieved: {result['retrieval_metadata']['num_chunks']}")
        print(f"  Level Distribution: {result['retrieval_metadata']['level_distribution']}")
        print(f"  Total Time: {result['total_time']:.2f}s")
        
        # Example 2: Generate quiz with hierarchical retrieval
        print("\n" + "="*70)
        print("EXAMPLE 2: Quiz Generation")
        print("="*70)
        
        quiz_result = tutor.generate_quiz(
            topic="Triangles",
            grade_level="Grade 7",
            num_questions=3,
            retrieval_strategy=RetrievalStrategy.HYBRID
        )
        
        if quiz_result['success']:
            print(f"\nQuiz on: {quiz_result.get('topic', 'N/A')}")
            print(f"Questions Generated: {len(quiz_result.get('quiz', {}).get('questions', []))}")
            print(f"Strategy Used: {quiz_result['metadata']['retrieval_strategy']}")
            print(f"Sources: {', '.join(quiz_result['metadata']['sources'][:3])}")
        
        # Example 3: Explain concept with hierarchical retrieval
        print("\n" + "="*70)
        print("EXAMPLE 3: Concept Explanation (Hierarchical)")
        print("="*70)
        
        explanation = tutor.explain_concept(
            concept="Similar triangles",
            grade_level="Grade 9",
            explanation_type="step-by-step",
            use_hierarchical=True
        )
        
        if explanation['success']:
            print(f"\nConcept: {explanation['concept']}")
            print(f"Explanation Preview: {explanation['explanation'][:200]}...")
            print(f"\nMetadata:")
            print(f"  Strategy: {explanation['metadata']['retrieval_strategy']}")
            print(f"  Level Distribution: {explanation['metadata']['level_distribution']}")
            print(f"  Chunks Used: {explanation['metadata']['num_chunks_used']}")
        
        # Example 4: Compare retrieval strategies
        print("\n" + "="*70)
        print("EXAMPLE 4: Strategy Comparison")
        print("="*70)
        
        comparison = tutor.compare_retrieval_strategies(
            query="What are the properties of a circle?",
            grade_level="Grade 8",
            top_k=5
        )
        
        print(f"\nQuery: {comparison['query']}")
        print("\nStrategy Comparison:")
        for strategy, data in comparison['strategies'].items():
            if data.get('success'):
                print(f"  {strategy}:")
                print(f"    Chunks: {data['num_chunks']}")
                print(f"    Avg Score: {data['avg_score']:.4f}")
                print(f"    Time: {data['retrieval_time']:.3f}s")
        
        print(f"\nRecommended Strategy: {comparison['recommendation']}")
        
        # Example 5: Get system statistics
        print("\n" + "="*70)
        print("EXAMPLE 5: System Statistics")
        print("="*70)
        
        stats = tutor.get_statistics()
        print(f"\nSystem Status: {stats['status']}")
        print(f"\nRAG Pipeline:")
        print(f"  Total Chunks: {stats['rag_pipeline']['total_chunks']}")
        print(f"  Embeddable: {stats['rag_pipeline']['embeddable_chunks']}")
        print(f"  Context-Only: {stats['rag_pipeline']['context_chunks']}")
        print(f"  Index Size: {stats['rag_pipeline']['index_size_mb']:.2f} MB")
        
        print(f"\nConfiguration:")
        print(f"  Default Top K: {stats['configuration']['default_top_k']}")
        print(f"  Max Context Tokens: {stats['configuration']['max_context_tokens']}")
        print(f"  Embedding Limit: {stats['configuration']['embedding_limit']}")
        print(f"  Reranker: {'Enabled' if stats['configuration']['reranker_enabled'] else 'Disabled'}")
        
        print(f"\nEnhanced Features:")
        for feature, enabled in stats['enhanced_features'].items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {feature.replace('_', ' ').title()}")
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Demo failed: {e}")
        print("  Please check that Elasticsearch is running and the index is populated.")