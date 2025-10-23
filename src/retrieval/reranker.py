# src/retrieval/reranker.py

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
import torch
import numpy as np

from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """Result from reranking operation."""
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]

class GeometryReranker:
    """Reranker for improving retrieval relevance."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Initialize reranker with specified model."""
        self.model_name = model_name
        self.device = settings.device if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading reranker model: {model_name} on {self.device}")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
        
        # Geometry-specific boost factors
        self.geometry_boosts = {
            'contains_theorem': 1.15,
            'contains_formula': 1.10,
            'contains_shape': 1.05,
            'high_topic_density': 1.08
        }
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        use_metadata_boost: bool = True,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3
    ) -> List[RerankResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Search query
            results: List of initial search results
            top_k: Number of top results to return (None = all)
            use_metadata_boost: Whether to apply geometry-specific boosts
            rerank_weight: Weight for reranker score
            original_weight: Weight for original retrieval score
        
        Returns:
            List of reranked results
        """
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = []
        for result in results:
            text = result['content'].get('text', '')
            pairs.append([query, text])
        
        # Get reranker scores
        try:
            rerank_scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Normalize scores to [0, 1]
            rerank_scores = self._normalize_scores(rerank_scores)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original scores
            rerank_scores = [result.get('score', 0) for result in results]
        
        # Combine scores and apply metadata boosts
        reranked_results = []
        
        for result, rerank_score in zip(results, rerank_scores):
            original_score = result.get('score', 0)
            
            # Normalize original score if needed
            if original_score > 1.0:
                original_score = original_score / max(r.get('score', 1) for r in results)
            
            # Combine scores
            combined_score = (
                rerank_weight * rerank_score + 
                original_weight * original_score
            )
            
            # Apply metadata boosts
            if use_metadata_boost:
                boost_factor = self._calculate_metadata_boost(
                    result['content']
                )
                combined_score *= boost_factor
            
            reranked_result = RerankResult(
                chunk_id=result['chunk_id'],
                text=result['content'].get('text', ''),
                original_score=original_score,
                rerank_score=rerank_score,
                final_score=combined_score,
                metadata=result['content']
            )
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Return top_k if specified
        if top_k:
            return reranked_results[:top_k]
        
        return reranked_results
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.ones_like(scores)
        
        return normalized
    
    def _calculate_metadata_boost(self, metadata: Dict[str, Any]) -> float:
        """Calculate boost factor based on geometry-specific metadata."""
        boost = 1.0
        
        # Boost for theorem content
        if metadata.get('contains_theorem', False):
            boost *= self.geometry_boosts['contains_theorem']
        
        # Boost for formula content
        if metadata.get('contains_formula', False):
            boost *= self.geometry_boosts['contains_formula']
        
        # Boost for shape-related content
        if metadata.get('contains_shape', False):
            boost *= self.geometry_boosts['contains_shape']
        
        # Boost for high topic density
        topic_density = metadata.get('topic_density', 0)
        if topic_density > 0.1:  # Threshold for "high density"
            boost *= self.geometry_boosts['high_topic_density']
        
        return boost
    
    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[RerankResult]]:
        """Rerank multiple query-result pairs in batch."""
        
        all_reranked = []
        
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k=top_k)
            all_reranked.append(reranked)
        
        return all_reranked


class GeometryMetadataScorer:
    """Additional scorer based on geometry-specific metadata."""
    
    def __init__(self):
        self.feature_weights = {
            'grade_level_match': 0.15,
            'difficulty_match': 0.10,
            'topic_relevance': 0.20,
            'content_type': 0.15,
            'recency': 0.10
        }
    
    def score_by_metadata(
        self,
        query_context: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Score results based on metadata alignment with query context.
        
        Args:
            query_context: Dictionary with query metadata
                - grade_level: Target grade level
                - difficulty: Desired difficulty
                - topics: Relevant topics
                - content_preference: Type of content needed
            
            results: Search results to score
        
        Returns:
            List of metadata scores
        """
        scores = []
        
        for result in results:
            content = result['content']
            score = 0.0
            
            # Grade level match
            if query_context.get('grade_level'):
                if content.get('grade_level') == query_context['grade_level']:
                    score += self.feature_weights['grade_level_match']
                elif self._is_adjacent_grade(
                    content.get('grade_level', ''),
                    query_context['grade_level']
                ):
                    score += self.feature_weights['grade_level_match'] * 0.5
            
            # Difficulty match
            if query_context.get('difficulty'):
                if content.get('difficulty') == query_context['difficulty']:
                    score += self.feature_weights['difficulty_match']
            
            # Topic relevance
            if query_context.get('topics'):
                content_topics = set(content.get('topics', []))
                query_topics = set(query_context['topics'])
                overlap = len(content_topics & query_topics)
                if query_topics:
                    topic_score = overlap / len(query_topics)
                    score += (
                        self.feature_weights['topic_relevance'] * topic_score
                    )
            
            # Content type preference
            if query_context.get('content_preference'):
                if self._matches_content_preference(
                    content,
                    query_context['content_preference']
                ):
                    score += self.feature_weights['content_type']
            
            scores.append(score)
        
        return scores
    
    def _is_adjacent_grade(self, grade1: str, grade2: str) -> bool:
        """Check if two grade levels are adjacent."""
        grade_order = [
            'Elementary (K-5)',
            'Middle School (6-8)',
            'High School (9-12)'
        ]
        
        try:
            idx1 = grade_order.index(grade1)
            idx2 = grade_order.index(grade2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False
    
    def _matches_content_preference(
        self,
        content: Dict[str, Any],
        preference: str
    ) -> bool:
        """Check if content matches the preferred type."""
        
        if preference == 'theorem':
            return content.get('contains_theorem', False)
        elif preference == 'formula':
            return content.get('contains_formula', False)
        elif preference == 'example':
            return 'example' in content.get('text', '').lower()
        elif preference == 'proof':
            return 'proof' in content.get('text', '').lower()
        
        return False
    
    def combine_scores(
        self,
        retrieval_scores: List[float],
        metadata_scores: List[float],
        retrieval_weight: float = 0.7,
        metadata_weight: float = 0.3
    ) -> List[float]:
        """
        Combine retrieval and metadata scores.
        
        Args:
            retrieval_scores: Scores from retrieval/reranking
            metadata_scores: Scores from metadata matching
            retrieval_weight: Weight for retrieval scores
            metadata_weight: Weight for metadata scores
        
        Returns:
            Combined scores
        """
        if len(retrieval_scores) != len(metadata_scores):
            raise ValueError("Score lists must have same length")
        
        combined = []
        for ret_score, meta_score in zip(retrieval_scores, metadata_scores):
            combined_score = (
                retrieval_weight * ret_score + 
                metadata_weight * meta_score
            )
            combined.append(combined_score)
        
        return combined


class AdaptiveReranker:
    """Adaptive reranker that learns from user feedback."""
    
    def __init__(self, base_reranker: GeometryReranker):
        self.base_reranker = base_reranker
        self.feedback_history = []
        self.boost_adjustments = {}
    
    def rerank_with_feedback(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_feedback: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank with consideration of user feedback.
        
        Args:
            query: Search query
            results: Initial search results
            user_feedback: Dictionary with feedback
                - relevant_chunks: List of chunk IDs marked as relevant
                - irrelevant_chunks: List of chunk IDs marked as irrelevant
                - preferred_difficulty: User's preferred difficulty level
            top_k: Number of results to return
        
        Returns:
            Reranked results adapted to feedback
        """
        
        # First, do standard reranking
        reranked = self.base_reranker.rerank(query, results, top_k=None)
        
        if not user_feedback:
            return reranked[:top_k] if top_k else reranked
        
        # Apply feedback-based adjustments
        relevant_chunks = set(user_feedback.get('relevant_chunks', []))
        irrelevant_chunks = set(user_feedback.get('irrelevant_chunks', []))
        
        for result in reranked:
            # Boost relevant chunks
            if result.chunk_id in relevant_chunks:
                result.final_score *= 1.3
            
            # Penalize irrelevant chunks
            if result.chunk_id in irrelevant_chunks:
                result.final_score *= 0.7
            
            # Adjust based on difficulty preference
            if user_feedback.get('preferred_difficulty'):
                pref_diff = user_feedback['preferred_difficulty']
                chunk_diff = result.metadata.get('difficulty', '')
                
                if chunk_diff == pref_diff:
                    result.final_score *= 1.1
        
        # Re-sort after adjustments
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Store feedback for future learning
        self.feedback_history.append({
            'query': query,
            'feedback': user_feedback,
            'timestamp': None  # Add timestamp if needed
        })
        
        return reranked[:top_k] if top_k else reranked
    
    def update_boost_factors(self):
        """
        Update boost factors based on accumulated feedback.
        This is a simplified version - in practice, you'd use more 
        sophisticated learning algorithms.
        """
        if len(self.feedback_history) < 10:
            return  # Need minimum feedback samples
        
        # Analyze feedback patterns
        # This is a placeholder for actual learning logic
        logger.info(f"Analyzed {len(self.feedback_history)} feedback samples")
        
        # In practice, you would:
        # 1. Analyze which metadata features correlate with relevance
        # 2. Adjust boost factors accordingly
        # 3. Use techniques like gradient descent or Bayesian optimization