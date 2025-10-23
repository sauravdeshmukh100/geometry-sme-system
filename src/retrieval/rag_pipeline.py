# src/retrieval/rag_pipeline.py

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..database.vector_store import GeometryVectorStore
from .reranker import GeometryReranker, GeometryMetadataScorer
from ..config.settings import settings

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Retrieval strategy options."""
    VECTOR_ONLY = "vector"
    KEYWORD_ONLY = "keyword"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"

@dataclass
class RetrievalConfig:
    """Configuration for RAG retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    rerank: bool = True
    rerank_top_k: int = 5
    use_metadata_boost: bool = True
    include_parents: bool = False
    include_children: bool = False
    level_preference: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    query: str
    chunks: List[Dict[str, Any]]
    context: str
    metadata: Dict[str, Any]
    config: RetrievalConfig

class GeometryRAGPipeline:
    """Complete RAG pipeline for geometry content retrieval."""
    
    def __init__(
        self,
        enable_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        """Initialize RAG pipeline components."""
        
        logger.info("Initializing Geometry RAG Pipeline")
        
        # Initialize vector store
        self.vector_store = GeometryVectorStore()
        
        # Initialize reranker
        self.reranker = None
        if enable_reranker:
            try:
                self.reranker = GeometryReranker(model_name=reranker_model)
                logger.info("Reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                logger.info("Continuing without reranker")
        
        # Initialize metadata scorer
        self.metadata_scorer = GeometryMetadataScorer()
        
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            config: Retrieval configuration
        
        Returns:
            RetrievalResult with chunks and assembled context
        """
        
        if config is None:
            config = RetrievalConfig()
        
        logger.info(f"Retrieving with strategy: {config.strategy.value}")
        
        # Step 1: Initial retrieval
        if config.strategy == RetrievalStrategy.VECTOR_ONLY:
            results = self.vector_store.vector_search(
                query,
                top_k=config.top_k * 2,  # Get more for reranking
                level=config.level_preference,
                filters=config.filters
            )
        elif config.strategy == RetrievalStrategy.KEYWORD_ONLY:
            results = self.vector_store.keyword_search(
                query,
                top_k=config.top_k * 2,
                level=config.level_preference,
                filters=config.filters
            )
        elif config.strategy == RetrievalStrategy.HYBRID:
            results = self.vector_store.hybrid_search(
                query,
                top_k=config.top_k * 2,
                level=config.level_preference,
                filters=config.filters
            )
        elif config.strategy == RetrievalStrategy.HIERARCHICAL:
            results = self._hierarchical_retrieval(query, config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        
        # Step 2: Rerank if enabled
        if config.rerank and self.reranker and results:
            logger.info("Reranking results")
            reranked = self.reranker.rerank(
                query,
                results,
                top_k=config.rerank_top_k,
                use_metadata_boost=config.use_metadata_boost
            )
            # Convert back to result format
            results = [
                {
                    'chunk_id': r.chunk_id,
                    'score': r.final_score,
                    'content': r.metadata
                }
                for r in reranked
            ]
        else:
            results = results[:config.rerank_top_k]
        
        # Step 3: Expand with parent/child chunks if requested
        if config.include_parents or config.include_children:
            results = self._expand_context(results, config)
        
        # Step 4: Assemble final context
        context = self._assemble_context(results)
        
        # Step 5: Prepare metadata
        metadata = self._prepare_metadata(results, query, config)
        
        return RetrievalResult(
            query=query,
            chunks=results,
            context=context,
            metadata=metadata,
            config=config
        )
    
    def _hierarchical_retrieval(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Perform hierarchical retrieval across all levels.
        
        Strategy:
        1. Search at Level 2 (fine-grained) for precise matches
        2. Get parent Level 1 chunks for context
        3. Optionally include Level 0 for broad context
        """
        
        # Search at fine-grained level (Level 2)
        l2_results = self.vector_store.hybrid_search(
            query,
            top_k=config.top_k,
            level=2,
            filters=config.filters
        )
        
        if not l2_results:
            # Fallback to Level 1
            return self.vector_store.hybrid_search(
                query,
                top_k=config.top_k,
                level=1,
                filters=config.filters
            )
        
        # Get parent chunks (Level 1)
        l2_chunk_ids = [r['chunk_id'] for r in l2_results]
        parent_chunks = self.vector_store.get_parent_chunks(l2_chunk_ids)
        
        # Combine results
        all_results = l2_results + parent_chunks
        
        # Deduplicate by chunk_id
        seen = set()
        unique_results = []
        for result in all_results:
            if result['chunk_id'] not in seen:
                seen.add(result['chunk_id'])
                unique_results.append(result)
        
        return unique_results[:config.top_k * 2]
    
    def _expand_context(
        self,
        results: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """Expand results with parent or child chunks."""
        
        expanded = list(results)
        chunk_ids = [r['chunk_id'] for r in results]
        
        if config.include_parents:
            parents = self.vector_store.get_parent_chunks(chunk_ids)
            expanded.extend(parents)
        
        if config.include_children:
            for chunk_id in chunk_ids:
                children = self.vector_store.get_children_chunks(chunk_id)
                expanded.extend(children)
        
        # Remove duplicates
        seen = set()
        unique = []
        for result in expanded:
            cid = result['chunk_id']
            if cid not in seen:
                seen.add(cid)
                unique.append(result)
        
        return unique
    
    def _assemble_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Assemble retrieved chunks into coherent context."""
        
        if not chunks:
            return ""
        
        # Sort by level (lower level = broader context first)
        # Then by position in document
        sorted_chunks = sorted(
            chunks,
            key=lambda x: (
                x['content'].get('level', 0),
                x['content'].get('doc_id', ''),
                x['content'].get('start_char', 0)
            )
        )
        
        context_parts = []
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk['content']
            text = content.get('text', '').strip()
            
            if not text:
                continue
            
            # Add source information for first chunk or when source changes
            if i == 0 or content.get('source') != sorted_chunks[i-1]['content'].get('source'):
                source = content.get('source', 'Unknown')
                grade_level = content.get('grade_level', 'General')
                difficulty = content.get('difficulty', 'Unknown')
                
                context_parts.append(
                    f"\n[Source: {source} | Grade: {grade_level} | "
                    f"Difficulty: {difficulty}]\n"
                )
            
            context_parts.append(text)
            context_parts.append("\n\n")
        
        return "".join(context_parts).strip()
    
    def _prepare_metadata(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        config: RetrievalConfig
    ) -> Dict[str, Any]:
        """Prepare metadata about retrieval results."""
        
        if not chunks:
            return {
                'num_chunks': 0,
                'sources': [],
                'topics': [],
                'grade_levels': [],
                'difficulties': []
            }
        
        sources = set()
        topics = set()
        grade_levels = set()
        difficulties = set()
        
        for chunk in chunks:
            content = chunk['content']
            
            if content.get('source'):
                sources.add(content['source'])
            
            if content.get('topics'):
                topics.update(content['topics'])
            
            if content.get('grade_level'):
                grade_levels.add(content['grade_level'])
            
            if content.get('difficulty'):
                difficulties.add(content['difficulty'])
        
        # Calculate average scores
        avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks)
        
        return {
            'num_chunks': len(chunks),
            'sources': list(sources),
            'topics': list(topics),
            'grade_levels': list(grade_levels),
            'difficulties': list(difficulties),
            'avg_score': avg_score,
            'strategy_used': config.strategy.value,
            'reranked': config.rerank
        }
    
    def retrieve_with_feedback(
        self,
        query: str,
        user_feedback: Optional[Dict[str, Any]] = None,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        Retrieve with user feedback for adaptive retrieval.
        
        Args:
            query: User query
            user_feedback: Feedback about previous results
                - preferred_grade_level: Preferred grade level
                - preferred_difficulty: Preferred difficulty
                - relevant_chunks: List of chunk IDs marked as relevant
                - irrelevant_chunks: List of chunk IDs marked as irrelevant
            config: Retrieval configuration
        """
        
        if config is None:
            config = RetrievalConfig()
        
        # Adjust filters based on feedback
        if user_feedback:
            if not config.filters:
                config.filters = {}
            
            if user_feedback.get('preferred_grade_level'):
                config.filters['grade_level'] = user_feedback['preferred_grade_level']
            
            if user_feedback.get('preferred_difficulty'):
                config.filters['difficulty'] = user_feedback['preferred_difficulty']
        
        # Perform retrieval
        result = self.retrieve(query, config)
        
        # If we have relevance feedback, re-rank based on similarity to relevant chunks
        if user_feedback and user_feedback.get('relevant_chunks'):
            result = self._rerank_with_relevance_feedback(
                result,
                user_feedback['relevant_chunks']
            )
        
        return result
    
    def _rerank_with_relevance_feedback(
        self,
        result: RetrievalResult,
        relevant_chunk_ids: List[str]
    ) -> RetrievalResult:
        """Re-rank results based on similarity to known relevant chunks."""
        
        # Get embeddings of relevant chunks
        relevant_chunks = []
        for chunk in result.chunks:
            if chunk['chunk_id'] in relevant_chunk_ids:
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            return result
        
        # Calculate similarity to relevant chunks and adjust scores
        for chunk in result.chunks:
            if chunk['chunk_id'] not in relevant_chunk_ids:
                # Boost score based on content similarity
                # (simplified - in practice, use actual embeddings)
                chunk['score'] *= 1.1  # Slight boost for similar content
        
        # Re-sort
        result.chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Reassemble context
        result.context = self._assemble_context(result.chunks)
        
        return result
    
    def batch_retrieve(
        self,
        queries: List[str],
        configs: Optional[List[RetrievalConfig]] = None
    ) -> List[RetrievalResult]:
        """Retrieve for multiple queries in batch."""
        
        if configs is None:
            configs = [RetrievalConfig() for _ in queries]
        
        results = []
        for query, config in zip(queries, configs):
            result = self.retrieve(query, config)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        
        # Get index statistics
        stats = self.vector_store.es_client.indices.stats(
            index=self.vector_store.index_name
        )
        
        index_stats = stats['indices'][self.vector_store.index_name]
        
        return {
            'total_chunks': index_stats['total']['docs']['count'],
            'index_size_mb': index_stats['total']['store']['size_in_bytes'] / (1024 * 1024),
            'embedding_model': settings.embedding_model,
            'reranker_enabled': self.reranker is not None,
            'cache_enabled': True
        }