# src/retrieval/reranker.py
"""
Enhanced Reranker for Geometry SME
Compatible with hierarchical retrieval and level-aware fusion
"""

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
    """Result from reranking operation with enhanced metadata."""
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]
    level: Optional[int] = None  # NEW: Chunk level (0, 1, 2)
    embeddable: bool = True  # NEW: Whether chunk is embeddable
    metadata_boost: float = 1.0  # NEW: Applied metadata boost factor

class GeometryReranker:
    """
    Enhanced reranker for improving retrieval relevance.
    
    Features:
    - Cross-encoder reranking
    - Geometry-specific metadata boosting
    - Level-aware scoring
    - Embeddable/context-only awareness
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Initialize reranker with specified model."""
        self.model_name = model_name
        self.device = settings.device if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading reranker model: {model_name} on {self.device}")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info("✓ Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load reranker model: {e}")
            raise
        
        # Enhanced geometry-specific boost factors
        self.geometry_boosts = {
            'contains_theorem': 1.15,
            'contains_formula': 1.10,
            'contains_shape': 1.05,
            'high_topic_density': 1.08,
            'contains_proof': 1.12,
            'has_examples': 1.06
        }
        
        # NEW: Level-specific boosts for hierarchical retrieval
        self.level_boosts = {
            0: 1.00,  # Context chunks (non-embeddable)
            1: 1.05,  # Medium chunks (≤384 tokens)
            2: 1.10   # Fine chunks (≤128 tokens) - prefer precision
        }
        
        # Track statistics
        self.rerank_count = 0
        self.avg_score_improvement = 0.0
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        use_metadata_boost: bool = True,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3,
        level_aware: bool = True  # NEW: Use level-aware boosting
    ) -> List[RerankResult]:
        """
        Rerank search results using cross-encoder with enhanced features.
        
        Args:
            query: Search query
            results: List of initial search results
            top_k: Number of top results to return (None = all)
            use_metadata_boost: Whether to apply geometry-specific boosts
            rerank_weight: Weight for reranker score (0.7 recommended)
            original_weight: Weight for original retrieval score (0.3 recommended)
            level_aware: Apply level-aware boosting for hierarchical retrieval
        
        Returns:
            List of reranked results
        """
        if not results:
            logger.warning("No results to rerank")
            return []
        
        logger.debug(f"Reranking {len(results)} results for query: {query[:50]}...")
        
        # Prepare pairs for reranking
        pairs = []
        for result in results:
            text = result['content'].get('text', '')
            if not text:
                logger.warning(f"Empty text for chunk {result.get('chunk_id', 'unknown')}")
                text = result['content'].get('title', '')  # Fallback
            pairs.append([query, text])
        
        # Get reranker scores
        try:
            rerank_scores = self.model.predict(
                pairs, 
                batch_size=32, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Normalize scores to [0, 1]
            rerank_scores = self._normalize_scores(rerank_scores)
            
            self.rerank_count += 1
            logger.debug(f"✓ Reranking complete: scores range [{rerank_scores.min():.3f}, {rerank_scores.max():.3f}]")
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback to original scores
            rerank_scores = np.array([result.get('score', 0) for result in results])
            rerank_scores = self._normalize_scores(rerank_scores)
        
        # Combine scores and apply metadata boosts
        reranked_results = []
        
        for result, rerank_score in zip(results, rerank_scores):
            original_score = result.get('score', 0)
            
            # Normalize original score if needed
            if original_score > 1.0:
                max_original = max(r.get('score', 1) for r in results)
                original_score = original_score / max_original if max_original > 0 else 0
            
            # Combine scores
            combined_score = (
                rerank_weight * rerank_score + 
                original_weight * original_score
            )
            
            # NEW: Apply level-aware boost
            level = result['content'].get('level', 1)
            embeddable = result['content'].get('embeddable', True)
            
            metadata_boost_factor = 1.0
            
            if level_aware and embeddable:
                level_boost = self.level_boosts.get(level, 1.0)
                combined_score *= level_boost
                metadata_boost_factor *= level_boost
            
            # Apply geometry-specific metadata boosts
            if use_metadata_boost:
                geo_boost = self._calculate_metadata_boost(result['content'])
                combined_score *= geo_boost
                metadata_boost_factor *= geo_boost
            
            # Create rerank result with enhanced metadata
            reranked_result = RerankResult(
                chunk_id=result['chunk_id'],
                text=result['content'].get('text', ''),
                original_score=original_score,
                rerank_score=float(rerank_score),
                final_score=combined_score,
                metadata=result['content'],
                level=level,
                embeddable=embeddable,
                metadata_boost=metadata_boost_factor
            )
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Calculate score improvement
        if len(results) > 0:
            orig_avg = np.mean([r.get('score', 0) for r in results])
            rerank_avg = np.mean([r.final_score for r in reranked_results])
            improvement = ((rerank_avg - orig_avg) / orig_avg * 100) if orig_avg > 0 else 0
            self.avg_score_improvement = (
                0.9 * self.avg_score_improvement + 0.1 * improvement
            )  # Running average
            logger.debug(f"Score improvement: {improvement:.1f}% (avg: {self.avg_score_improvement:.1f}%)")
        
        # Return top_k if specified
        if top_k:
            return reranked_results[:top_k]
        
        return reranked_results
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: Array of scores to normalize
        
        Returns:
            Normalized scores
        """
        scores = np.array(scores, dtype=np.float32)
        
        # Handle edge cases
        if len(scores) == 0:
            return scores
        
        if len(scores) == 1:
            return np.array([1.0])
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 1e-6:  # Avoid division by zero
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            # All scores are the same
            normalized = np.ones_like(scores) * 0.5
        
        return normalized
    
    def _calculate_metadata_boost(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate boost factor based on geometry-specific metadata.
        
        Enhanced to include more content indicators.
        
        Args:
            metadata: Chunk metadata
        
        Returns:
            Boost factor (≥ 1.0)
        """
        boost = 1.0
        
        # Boost for theorem content
        if metadata.get('contains_theorem', False):
            boost *= self.geometry_boosts['contains_theorem']
            logger.debug("  Applied theorem boost")
        
        # Boost for formula content
        if metadata.get('contains_formula', False):
            boost *= self.geometry_boosts['contains_formula']
            logger.debug("  Applied formula boost")
        
        # Boost for shape-related content
        if metadata.get('contains_shape', False):
            boost *= self.geometry_boosts['contains_shape']
            logger.debug("  Applied shape boost")
        
        # NEW: Boost for proof content
        if metadata.get('contains_proof', False):
            boost *= self.geometry_boosts['contains_proof']
            logger.debug("  Applied proof boost")
        
        # NEW: Boost for examples
        text = metadata.get('text', '').lower()
        if 'example' in text or 'for instance' in text:
            boost *= self.geometry_boosts['has_examples']
            logger.debug("  Applied example boost")
        
        # Boost for high topic density
        topic_density = metadata.get('topic_density', 0)
        if topic_density > 0.1:  # Threshold for "high density"
            boost *= self.geometry_boosts['high_topic_density']
            logger.debug(f"  Applied topic density boost ({topic_density:.2f})")
        
        # Cap maximum boost to prevent over-boosting
        boost = min(boost, 1.5)
        
        return boost
    
    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[List[RerankResult]]:
        """
        Rerank multiple query-result pairs in batch.
        
        Args:
            queries: List of queries
            results_list: List of result lists (one per query)
            top_k: Number of results per query
            **kwargs: Additional arguments for rerank()
        
        Returns:
            List of reranked results (one list per query)
        """
        logger.info(f"Batch reranking {len(queries)} queries")
        
        all_reranked = []
        
        for i, (query, results) in enumerate(zip(queries, results_list)):
            logger.debug(f"Reranking batch item {i+1}/{len(queries)}")
            reranked = self.rerank(query, results, top_k=top_k, **kwargs)
            all_reranked.append(reranked)
        
        logger.info(f"✓ Batch reranking complete")
        return all_reranked
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            'model': self.model_name,
            'device': self.device,
            'rerank_count': self.rerank_count,
            'avg_score_improvement': self.avg_score_improvement,
            'geometry_boosts': self.geometry_boosts,
            'level_boosts': self.level_boosts
        }


class GeometryMetadataScorer:
    """
    Enhanced metadata scorer for geometry-specific content.
    
    Scores based on:
    - Grade level alignment
    - Difficulty matching
    - Topic relevance
    - Content type preferences
    - Level appropriateness (NEW)
    """
    
    def __init__(self):
        # Feature weights (should sum to ~1.0)
        self.feature_weights = {
            'grade_level_match': 0.15,
            'difficulty_match': 0.10,
            'topic_relevance': 0.20,
            'content_type': 0.15,
            'level_appropriateness': 0.10,  # NEW
            'recency': 0.05,
            'source_quality': 0.10,  # NEW
            'completeness': 0.15  # NEW
        }
        
        # Grade level mapping for adjacency
        self.grade_hierarchy = {
            'Grade 6': 6, 'Grade 7': 7, 'Grade 8': 8,
            'Grade 9': 9, 'Grade 10': 10
        }
        
        logger.info("Metadata Scorer initialized")
    
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
                - preferred_level: Preferred chunk level (0, 1, 2)
            
            results: Search results to score
        
        Returns:
            List of metadata scores (0.0 to 1.0)
        """
        scores = []
        
        for result in results:
            content = result['content']
            score = 0.0
            
            # 1. Grade level match
            if query_context.get('grade_level'):
                grade_score = self._score_grade_match(
                    content.get('grade_level', ''),
                    query_context['grade_level']
                )
                score += self.feature_weights['grade_level_match'] * grade_score
            
            # 2. Difficulty match
            if query_context.get('difficulty'):
                if content.get('difficulty') == query_context['difficulty']:
                    score += self.feature_weights['difficulty_match']
                elif self._is_adjacent_difficulty(
                    content.get('difficulty', ''),
                    query_context['difficulty']
                ):
                    score += self.feature_weights['difficulty_match'] * 0.6
            
            # 3. Topic relevance
            if query_context.get('topics'):
                topic_score = self._score_topic_relevance(
                    content.get('topics', []),
                    query_context['topics']
                )
                score += self.feature_weights['topic_relevance'] * topic_score
            
            # 4. Content type preference
            if query_context.get('content_preference'):
                if self._matches_content_preference(
                    content,
                    query_context['content_preference']
                ):
                    score += self.feature_weights['content_type']
            
            # 5. NEW: Level appropriateness
            if query_context.get('preferred_level') is not None:
                level_score = self._score_level_match(
                    content.get('level', 1),
                    query_context['preferred_level']
                )
                score += self.feature_weights['level_appropriateness'] * level_score
            
            # 6. NEW: Source quality
            source_quality = self._assess_source_quality(content)
            score += self.feature_weights['source_quality'] * source_quality
            
            # 7. NEW: Content completeness
            completeness = self._assess_completeness(content)
            score += self.feature_weights['completeness'] * completeness
            
            scores.append(min(score, 1.0))  # Cap at 1.0
        
        return scores
    
    def _score_grade_match(self, content_grade: str, target_grade: str) -> float:
        """Score grade level match with adjacency consideration."""
        if content_grade == target_grade:
            return 1.0
        
        if self._is_adjacent_grade(content_grade, target_grade):
            return 0.6
        
        # Calculate distance penalty
        c_num = self.grade_hierarchy.get(content_grade, 0)
        t_num = self.grade_hierarchy.get(target_grade, 0)
        
        if c_num and t_num:
            distance = abs(c_num - t_num)
            return max(0, 1.0 - (distance * 0.2))
        
        return 0.3  # Unknown grade
    
    def _is_adjacent_grade(self, grade1: str, grade2: str) -> bool:
        """Check if two grade levels are adjacent."""
        g1_num = self.grade_hierarchy.get(grade1)
        g2_num = self.grade_hierarchy.get(grade2)
        
        if g1_num and g2_num:
            return abs(g1_num - g2_num) == 1
        
        return False
    
    def _is_adjacent_difficulty(self, diff1: str, diff2: str) -> bool:
        """Check if two difficulty levels are adjacent."""
        difficulty_order = ['Beginner', 'Intermediate', 'Advanced']
        
        try:
            idx1 = difficulty_order.index(diff1)
            idx2 = difficulty_order.index(diff2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False
    
    def _score_topic_relevance(
        self,
        content_topics: List[str],
        query_topics: List[str]
    ) -> float:
        """Score topic overlap using Jaccard similarity."""
        if not query_topics:
            return 0.5  # Neutral
        
        content_set = set(t.lower() for t in content_topics)
        query_set = set(t.lower() for t in query_topics)
        
        if not content_set:
            return 0.0
        
        intersection = len(content_set & query_set)
        union = len(content_set | query_set)
        
        return intersection / union if union > 0 else 0.0
    
    def _matches_content_preference(
        self,
        content: Dict[str, Any],
        preference: str
    ) -> bool:
        """Check if content matches the preferred type."""
        preference_lower = preference.lower()
        
        if preference_lower == 'theorem':
            return content.get('contains_theorem', False)
        elif preference_lower == 'formula':
            return content.get('contains_formula', False)
        elif preference_lower == 'example':
            text = content.get('text', '').lower()
            return 'example' in text or 'for instance' in text
        elif preference_lower == 'proof':
            return content.get('contains_proof', False) or 'proof' in content.get('text', '').lower()
        elif preference_lower == 'definition':
            text = content.get('text', '').lower()
            return any(kw in text for kw in ['define', 'definition', 'is defined as'])
        
        return False
    
    def _score_level_match(self, content_level: int, preferred_level: int) -> float:
        """Score how well the content level matches preference."""
        if content_level == preferred_level:
            return 1.0
        
        # Adjacent levels get partial credit
        if abs(content_level - preferred_level) == 1:
            return 0.6
        
        return 0.3
    
    def _assess_source_quality(self, content: Dict[str, Any]) -> float:
        """Assess quality of the source document."""
        score = 0.5  # Base score
        
        source = content.get('source', '').lower()
        
        # Higher quality for textbooks and official materials
        if any(kw in source for kw in ['textbook', 'curriculum', 'ncert', 'khan']):
            score += 0.3
        
        # Check for authoritative markers
        if content.get('has_citations', False):
            score += 0.1
        
        # Check for structured content
        if content.get('contains_theorem') or content.get('contains_formula'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_completeness(self, content: Dict[str, Any]) -> float:
        """Assess completeness of the content."""
        score = 0.5  # Base score
        
        text_length = len(content.get('text', ''))
        
        # Prefer moderate length (not too short, not too long)
        if 100 < text_length < 500:
            score += 0.3
        elif 50 < text_length < 100:
            score += 0.1
        
        # Check for multiple content types
        has_types = sum([
            content.get('contains_theorem', False),
            content.get('contains_formula', False),
            content.get('contains_shape', False),
            'example' in content.get('text', '').lower()
        ])
        
        if has_types >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def combine_scores(
        self,
        retrieval_scores: List[float],
        metadata_scores: List[float],
        retrieval_weight: float = 0.7,
        metadata_weight: float = 0.3
    ) -> List[float]:
        """
        Combine retrieval and metadata scores with weights.
        
        Args:
            retrieval_scores: Scores from retrieval/reranking
            metadata_scores: Scores from metadata matching
            retrieval_weight: Weight for retrieval scores
            metadata_weight: Weight for metadata scores
        
        Returns:
            Combined scores
        """
        if len(retrieval_scores) != len(metadata_scores):
            raise ValueError(f"Score lists must have same length: {len(retrieval_scores)} vs {len(metadata_scores)}")
        
        # Normalize weights
        total_weight = retrieval_weight + metadata_weight
        if total_weight > 0:
            retrieval_weight /= total_weight
            metadata_weight /= total_weight
        
        combined = []
        for ret_score, meta_score in zip(retrieval_scores, metadata_scores):
            combined_score = (
                retrieval_weight * ret_score + 
                metadata_weight * meta_score
            )
            combined.append(combined_score)
        
        return combined


class AdaptiveReranker:
    """
    Adaptive reranker that learns from user feedback.
    
    Features:
    - Feedback-based score adjustments
    - Learning from relevance judgments
    - Difficulty preference adaptation
    """
    
    def __init__(self, base_reranker: GeometryReranker):
        self.base_reranker = base_reranker
        self.feedback_history = []
        self.boost_adjustments = {}
        self.difficulty_preferences = {}
        
        logger.info("Adaptive Reranker initialized")
    
    def rerank_with_feedback(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_feedback: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank with consideration of user feedback and preferences.
        
        Args:
            query: Search query
            results: Initial search results
            user_feedback: Dictionary with feedback
                - relevant_chunks: List of chunk IDs marked as relevant
                - irrelevant_chunks: List of chunk IDs marked as irrelevant
                - preferred_difficulty: User's preferred difficulty level
                - preferred_level: Preferred chunk level
            user_id: Optional user identifier for personalization
            top_k: Number of results to return
        
        Returns:
            Reranked results adapted to feedback
        """
        
        # First, do standard reranking
        reranked = self.base_reranker.rerank(query, results, top_k=None)
        
        if not user_feedback:
            return reranked[:top_k] if top_k else reranked
        
        logger.info(f"Applying feedback-based adjustments for user {user_id}")
        
        # Apply feedback-based adjustments
        relevant_chunks = set(user_feedback.get('relevant_chunks', []))
        irrelevant_chunks = set(user_feedback.get('irrelevant_chunks', []))
        
        for result in reranked:
            # Boost relevant chunks
            if result.chunk_id in relevant_chunks:
                result.final_score *= 1.3
                logger.debug(f"  Boosted relevant chunk: {result.chunk_id}")
            
            # Penalize irrelevant chunks
            if result.chunk_id in irrelevant_chunks:
                result.final_score *= 0.7
                logger.debug(f"  Penalized irrelevant chunk: {result.chunk_id}")
            
            # Adjust based on difficulty preference
            if user_feedback.get('preferred_difficulty'):
                pref_diff = user_feedback['preferred_difficulty']
                chunk_diff = result.metadata.get('difficulty', '')
                
                if chunk_diff == pref_diff:
                    result.final_score *= 1.1
                    logger.debug(f"  Boosted preferred difficulty: {chunk_diff}")
            
            # NEW: Adjust based on level preference
            if user_feedback.get('preferred_level') is not None:
                pref_level = user_feedback['preferred_level']
                chunk_level = result.level
                
                if chunk_level == pref_level:
                    result.final_score *= 1.08
                    logger.debug(f"  Boosted preferred level: {chunk_level}")
        
        # Re-sort after adjustments
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Store feedback for future learning
        self.feedback_history.append({
            'query': query,
            'feedback': user_feedback,
            'user_id': user_id,
            'timestamp': self._get_timestamp()
        })
        
        # Update user preferences if user_id provided
        if user_id and user_feedback.get('preferred_difficulty'):
            self.difficulty_preferences[user_id] = user_feedback['preferred_difficulty']
        
        logger.info(f"✓ Applied feedback adjustments, {len(self.feedback_history)} total feedback samples")
        
        return reranked[:top_k] if top_k else reranked
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for a user."""
        return {
            'preferred_difficulty': self.difficulty_preferences.get(user_id),
            'feedback_count': sum(
                1 for f in self.feedback_history 
                if f.get('user_id') == user_id
            )
        }
    
    def update_boost_factors(self):
        """
        Update boost factors based on accumulated feedback.
        
        This is a simplified version - in production, you'd use more 
        sophisticated learning algorithms (e.g., learning-to-rank).
        """
        if len(self.feedback_history) < 10:
            logger.info("Not enough feedback samples for learning")
            return
        
        logger.info(f"Analyzing {len(self.feedback_history)} feedback samples")
        
        # Analyze difficulty preferences
        difficulty_counts = {}
        for feedback in self.feedback_history:
            pref_diff = feedback['feedback'].get('preferred_difficulty')
            if pref_diff:
                difficulty_counts[pref_diff] = difficulty_counts.get(pref_diff, 0) + 1
        
        if difficulty_counts:
            logger.info(f"Difficulty preferences: {difficulty_counts}")
        
        # In production, you would:
        # 1. Analyze which features correlate with relevance
        # 2. Adjust boost factors using gradient descent or similar
        # 3. Use techniques like ListNet, LambdaMART for learning-to-rank
        # 4. A/B test different boost configurations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive reranker statistics."""
        return {
            **self.base_reranker.get_statistics(),
            'feedback_samples': len(self.feedback_history),
            'unique_users': len(set(
                f.get('user_id') for f in self.feedback_history 
                if f.get('user_id')
            )),
            'difficulty_preferences': self.difficulty_preferences.copy()
        }