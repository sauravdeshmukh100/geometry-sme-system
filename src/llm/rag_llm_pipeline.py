# src/llm/rag_llm_pipeline.py

from typing import Dict, Any, Optional, List
import logging
import os

from ..retrieval.rag_pipeline import GeometryRAGPipeline, RetrievalConfig, RetrievalStrategy
from .gemini_client import GeminiClient
from ..config.settings import settings

# Setup logging
log_file = getattr(settings, 'log_file', os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'rag_llm_pipeline.log'))
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
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the complete tutor pipeline.
        
        Args:
            gemini_api_key: Google Gemini API key
        """
        logger.info("Initializing Geometry Tutor Pipeline")
        
        # Initialize RAG pipeline
        self.rag_pipeline = GeometryRAGPipeline(enable_reranker=True)
        logger.info("✓ RAG pipeline initialized")
        gemini_api_key = settings.GEMINI_API_KEY if gemini_api_key is None else gemini_api_key
        # Initialize LLM client
        print("gemini_api_key:", gemini_api_key)
        self.llm_client = GeminiClient(api_key=gemini_api_key)
        logger.info("✓ Gemini LLM initialized")
        
        # Test connections
        self._test_system()
    
    def answer_question(
        self,
        query: str,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a geometry question using RAG + LLM.
        
        Args:
            query: Student's question
            grade_level: Target grade level (e.g., "Grade 7")
            difficulty: Content difficulty
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source references
        
        Returns:
            Dict with answer, sources, and metadata
        """
        
        logger.info(f"Answering question: {query}")
        
        # Step 1: Retrieve relevant context
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=top_k * 2,
            rerank=True,
            rerank_top_k=top_k,
            filters={'grade_level': grade_level} if grade_level else None
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(query, retrieval_config)
            logger.info(f"Retrieved {len(retrieval_result.chunks)} relevant chunks")
            
            if not retrieval_result.chunks:
                return {
                    'answer': "I couldn't find relevant information in my knowledge base. Could you rephrase your question?",
                    'sources': [],
                    'success': False,
                    'error': 'No relevant context found'
                }
            
            # Step 2: Generate answer using LLM
            llm_response = self.llm_client.generate_answer(
                query=query,
                context=retrieval_result.context,
                grade_level=grade_level or retrieval_result.metadata.get('grade_levels', [None])[0],
                difficulty=difficulty or retrieval_result.metadata.get('difficulties', [None])[0]
            )
            
            # Step 3: Compile response
            response = {
                'query': query,
                'answer': llm_response['answer'],
                'context_used': retrieval_result.context[:500] + "..." if len(retrieval_result.context) > 500 else retrieval_result.context,
                'retrieval_metadata': {
                    'num_chunks': len(retrieval_result.chunks),
                    'sources': retrieval_result.metadata.get('sources', []),
                    'grade_levels': retrieval_result.metadata.get('grade_levels', []),
                    'avg_score': retrieval_result.metadata.get('avg_score', 0)
                },
                'generation_time': llm_response.get('generation_time', 0),
                'success': llm_response.get('success', False)
            }
            
            # Add source references if requested
            if include_sources:
                response['sources'] = [
                    {
                        'source': chunk['content'].get('source'),
                        'grade': chunk['content'].get('grade_level'),
                        'score': chunk.get('score', 0)
                    }
                    for chunk in retrieval_result.chunks[:3]
                ]
            
            logger.info("✓ Answer generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                'answer': f"An error occurred: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def generate_quiz(
        self,
        topic: str,
        grade_level: str,
        num_questions: int = 5,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a quiz on a specific geometry topic.
        
        Args:
            topic: Topic for quiz (e.g., "Triangles", "Circles")
            grade_level: Target grade level
            num_questions: Number of questions
            difficulty: Optional difficulty filter
        
        Returns:
            Dict with quiz content and metadata
        """
        
        logger.info(f"Generating quiz on: {topic} for {grade_level}")
        
        # Step 1: Retrieve context about the topic
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=10,
            rerank=True,
            rerank_top_k=5,
            filters={
                'grade_level': grade_level,
                'difficulty': difficulty
            } if difficulty else {'grade_level': grade_level}
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(
                f"Explain {topic} concepts, formulas, and theorems",
                retrieval_config
            )
            
            if not retrieval_result.chunks:
                return {
                    'quiz': None,
                    'success': False,
                    'error': f'No content found for topic: {topic}'
                }
            
            # Step 2: Generate quiz using LLM
            quiz_response = self.llm_client.generate_quiz(
                topic=topic,
                context=retrieval_result.context,
                grade_level=grade_level,
                num_questions=num_questions
            )
            
            # Step 3: Compile response
            response = {
                'quiz': quiz_response['quiz'],
                'topic': topic,
                'grade_level': grade_level,
                'num_questions': num_questions,
                'sources': retrieval_result.metadata.get('sources', []),
                'generated_at': quiz_response.get('generated_at'),
                'success': quiz_response.get('success', False)
            }
            
            logger.info("✓ Quiz generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_quiz: {e}", exc_info=True)
            return {
                'quiz': None,
                'success': False,
                'error': str(e)
            }
    
    def explain_concept(
        self,
        concept: str,
        grade_level: str,
        explanation_type: str = "step-by-step"
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a concept.
        
        Args:
            concept: Concept to explain
            grade_level: Target grade level
            explanation_type: Type of explanation
        
        Returns:
            Dict with explanation and metadata
        """
        
        logger.info(f"Explaining: {concept} ({explanation_type})")
        
        # Retrieve context
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.HIERARCHICAL,
            top_k=8,
            rerank=True,
            include_parents=True,
            filters={'grade_level': grade_level}
        )
        
        try:
            retrieval_result = self.rag_pipeline.retrieve(concept, retrieval_config)
            
            if not retrieval_result.chunks:
                return {
                    'explanation': f"I don't have information about '{concept}' in my knowledge base.",
                    'success': False
                }
            
            # Generate explanation
            explanation_response = self.llm_client.generate_explanation(
                concept=concept,
                context=retrieval_result.context,
                grade_level=grade_level,
                explanation_type=explanation_type
            )
            
            return {
                'concept': concept,
                'explanation': explanation_response['explanation'],
                'grade_level': grade_level,
                'type': explanation_type,
                'sources': retrieval_result.metadata.get('sources', []),
                'success': explanation_response.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"Error in explain_concept: {e}", exc_info=True)
            return {
                'explanation': None,
                'success': False,
                'error': str(e)
            }
    
    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieve_context: bool = True,
        grade_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle conversational interaction with memory and context.
        
        Args:
            message: User's message
            chat_history: Previous conversation
            retrieve_context: Whether to retrieve relevant context
            grade_level: Optional grade filter
        
        Returns:
            Dict with response and updated history
        """
        
        logger.info(f"Chat message: {message}")
        
        context = None
        
        # Retrieve context if requested
        if retrieve_context:
            try:
                retrieval_config = RetrievalConfig(
                    strategy=RetrievalStrategy.HYBRID,
                    top_k=3,
                    rerank=True,
                    filters={'grade_level': grade_level} if grade_level else None
                )
                
                retrieval_result = self.rag_pipeline.retrieve(message, retrieval_config)
                
                if retrieval_result.chunks:
                    context = retrieval_result.context[:1000]  # Limit context
                    
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
        
        # Generate response
        chat_response = self.llm_client.chat(
            message=message,
            chat_history=chat_history or [],
            context=context
        )
        
        return chat_response
    
    def _test_system(self):
        """Test that all components are working."""
        logger.info("Testing system components...")
        
        # Test RAG
        try:
            test_result = self.rag_pipeline.get_statistics()
            logger.info(f"✓ RAG system ready: {test_result.get('total_chunks', 0)} chunks indexed")
        except Exception as e:
            logger.error(f"✗ RAG system test failed: {e}")
        
        # Test LLM
        try:
            self.llm_client.test_connection()
        except Exception as e:
            logger.error(f"✗ LLM test failed: {e}")
        
        logger.info("System test complete")