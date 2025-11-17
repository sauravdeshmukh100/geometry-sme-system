# src/retrieval/rag_pipeline.py
"""
Enhanced RAG Pipeline with improved hierarchical retrieval,
adaptive chunk selection, and better context assembly.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..database.vector_store import GeometryVectorStore
from .reranker import GeometryReranker, GeometryMetadataScorer
from ..config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Retrieval strategy options."""
    VECTOR_ONLY = "vector"
    KEYWORD_ONLY = "keyword"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"  # NEW: Auto-select strategy

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
    embeddable_only: bool = True
    max_context_tokens: int = 4000
    enable_deduplication: bool = True
    similarity_threshold: float = 0.95

@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    query: str
    chunks: List[Dict[str, Any]]
    context: str
    metadata: Dict[str, Any]
    config: RetrievalConfig

class GeometryRAGPipeline:
    """
    Enhanced RAG pipeline for geometry content retrieval.
    
    Improvements:
    - Better hierarchical retrieval with level-aware fusion
    - Adaptive chunk selection based on query type
    - Enhanced deduplication
    - Token-aware context assembly
    """
    
    def __init__(
        self,
        enable_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        """Initialize RAG pipeline components."""
        
        logger.info("Initializing Enhanced Geometry RAG Pipeline")
        
        # Initialize vector store
        self.vector_store = GeometryVectorStore()
        
        # Initialize reranker
        self.reranker = None
        if enable_reranker:
            try:
                self.reranker = GeometryReranker(model_name=reranker_model)
                logger.info("✓ Reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                logger.info("Continuing without reranker")
        
        # Initialize metadata scorer
        self.metadata_scorer = GeometryMetadataScorer()
        
        # Get embedding model limit from settings
        self.embedding_limit = getattr(settings, 'EMBEDDING_MAX_TOKENS', 384)
        logger.info(f"✓ Embedding model limit: {self.embedding_limit} tokens")
        
        # Query type detection patterns
        self._init_query_patterns()
        
    def _init_query_patterns(self):
        """Initialize query type detection patterns."""
        self.query_patterns = {
            'factual': ['what is', 'define', 'definition', 'meaning of', 'formula for'],
            'explanation': ['explain', 'how', 'why', 'describe', 'tell me about'],
            'complex': ['solve', 'prove', 'calculate', 'derive', 'demonstrate', 'show that'],
            'listing': ['list', 'types of', 'kinds of', 'properties of', 'examples of']
        }
    
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        Enhanced retrieval with adaptive strategy selection.
        
        Args:
            query: User query
            config: Retrieval configuration
        
        Returns:
            RetrievalResult with chunks and assembled context
        """
        
        if config is None:
            config = RetrievalConfig()
        
        # ADAPTIVE: Auto-select strategy if requested
        if config.strategy == RetrievalStrategy.ADAPTIVE:
            config.strategy = self._select_optimal_strategy(query)
            logger.info(f"✓ Auto-selected strategy: {config.strategy.value}")
        
        # ADAPTIVE: Auto-select level if not specified
        if config.level_preference is None and config.strategy != RetrievalStrategy.HIERARCHICAL:
            config.level_preference = self._select_optimal_level(query)
            logger.info(f"✓ Auto-selected level: {config.level_preference}")
        
        logger.info(f"Retrieving with strategy: {config.strategy.value}")
        
        # Add embeddable filter if specified
        if config.embeddable_only:
            if config.filters is None:
                config.filters = {}
            config.filters['embeddable'] = True
        
        # Step 1: Initial retrieval based on strategy
        results = self._execute_retrieval_strategy(query, config)
        
        # Step 2: Rerank if enabled
        if config.rerank and self.reranker and results:
            results = self._rerank_results(query, results, config)
        else:
            results = results[:config.rerank_top_k]
        
        # Step 3: Expand with parent/child chunks if requested
        if config.include_parents or config.include_children:
            results = self._expand_context(results, config)
        
        # Step 4: Deduplicate if enabled
        if config.enable_deduplication:
            results = self._deduplicate_chunks(results, config.similarity_threshold)
        
        # Step 5: Assemble final context with token awareness
        context = self._assemble_context(results, config.max_context_tokens)
        
        # Step 6: Prepare metadata
        metadata = self._prepare_metadata(results, query, config)
        
        return RetrievalResult(
            query=query,
            chunks=results,
            context=context,
            metadata=metadata,
            config=config
        )
    
    def _execute_retrieval_strategy(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """Execute retrieval based on selected strategy."""
        
        if config.strategy == RetrievalStrategy.VECTOR_ONLY:
            return self.vector_store.vector_search(
                query,
                top_k=config.top_k * 2,
                level=config.level_preference,
                filters=config.filters
            )
        
        elif config.strategy == RetrievalStrategy.KEYWORD_ONLY:
            return self.vector_store.keyword_search(
                query,
                top_k=config.top_k * 2,
                level=config.level_preference,
                filters=config.filters
            )
        
        elif config.strategy == RetrievalStrategy.HYBRID:
            return self.vector_store.hybrid_search(
                query,
                top_k=config.top_k * 2,
                level=config.level_preference,
                filters=config.filters
            )
        
        elif config.strategy == RetrievalStrategy.HIERARCHICAL:
            return self._hierarchical_retrieval(query, config)
        
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _hierarchical_retrieval(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        ENHANCED hierarchical retrieval with multi-stage fusion.
        
        Strategy:
        1. Search L2 (fine-grained, ≤128 tokens) for precise matches
        2. Search L1 (medium, ≤384 tokens) for contextual matches
        3. Fuse results with level-aware scoring
        4. Optionally expand with L0 context chunks
        """
        
        logger.info("Executing enhanced hierarchical retrieval")
        
        # Stage 1: Search both embeddable levels independently
        l2_results = self.vector_store.hybrid_search(
            query,
            top_k=config.top_k,
            level=2,
            filters=config.filters
        )
        
        l1_results = self.vector_store.hybrid_search(
            query,
            top_k=config.top_k,
            level=1,
            filters=config.filters
        )
        
        logger.info(f"Retrieved: L2={len(l2_results)}, L1={len(l1_results)}")
        
        # Stage 2: Fuse with level-aware scoring
        # L2 provides precision, L1 provides context
        fused_results = self._level_aware_fusion(
            l2_results, 
            l1_results,
            l2_weight=0.6,  # Prefer precision
            l1_weight=0.4   # But maintain context
        )
        
        # Stage 3: Optionally get L0 context chunks (non-embeddable)
        if config.include_parents:
            chunk_ids = [r['chunk_id'] for r in fused_results[:config.top_k]]
            l0_context = self.vector_store.get_parent_chunks(chunk_ids)
            
            # Add L0 chunks with lower scores (they're context-only)
            for l0_chunk in l0_context:
                l0_chunk['score'] = 0.3  # Lower score for context chunks
            
            fused_results.extend(l0_context)
            logger.info(f"Added {len(l0_context)} L0 context chunks")
        
        return fused_results[:config.top_k * 2]
    
    def _level_aware_fusion(
        self,
        l2_results: List[Dict],
        l1_results: List[Dict],
        l2_weight: float = 0.6,
        l1_weight: float = 0.4
    ) -> List[Dict]:
        """
        Fuse results from different levels with level-aware scoring.
        
        This prevents L1 and L2 chunks from competing directly,
        instead leveraging the strengths of each level.
        """
        
        # Normalize scores within each level
        def normalize(results):
            if not results:
                return []
            scores = [r['score'] for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            
            for r in results:
                r['normalized_score'] = (r['score'] - min_s) / range_s
            return results
        
        l2_results = normalize(l2_results)
        l1_results = normalize(l1_results)
        
        # Merge with weights
        chunk_scores = {}
        
        # Add L2 results (precision-focused)
        for r in l2_results:
            cid = r['chunk_id']
            chunk_scores[cid] = {
                'content': r['content'],
                'score': r['normalized_score'] * l2_weight,
                'level': 2,
                'from_l2': True,
                'from_l1': False
            }
        
        # Add L1 results (context-focused)
        for r in l1_results:
            cid = r['chunk_id']
            if cid in chunk_scores:
                # Chunk appears in both levels - significant boost!
                chunk_scores[cid]['score'] += r['normalized_score'] * l1_weight
                chunk_scores[cid]['from_l1'] = True
            else:
                chunk_scores[cid] = {
                    'content': r['content'],
                    'score': r['normalized_score'] * l1_weight,
                    'level': 1,
                    'from_l2': False,
                    'from_l1': True
                }
        
        # Sort by final score
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Format back to standard structure
        fused = []
        for cid, data in sorted_results:
            fused.append({
                'chunk_id': cid,
                'score': data['score'],
                'content': data['content'],
                'metadata': {
                    'fusion_level': data['level'],
                    'in_both_levels': data['from_l2'] and data['from_l1']
                }
            })
        
        logger.info(f"Fused into {len(fused)} unique chunks")
        return fused
    
    def _select_optimal_strategy(self, query: str) -> RetrievalStrategy:
        """
        Select optimal retrieval strategy based on query characteristics.
        """
        query_lower = query.lower()
        
        # Complex multi-step queries → Hierarchical
        if any(kw in query_lower for kw in self.query_patterns['complex']):
            return RetrievalStrategy.HIERARCHICAL
        
        # Listing/enumeration queries → Keyword works well
        if any(kw in query_lower for kw in self.query_patterns['listing']):
            return RetrievalStrategy.HYBRID  # Hybrid still better than pure keyword
        
        # Default: Hybrid (best overall)
        return RetrievalStrategy.HYBRID
    
    def _select_optimal_level(self, query: str) -> Optional[int]:
        """
        Select optimal retrieval level based on query characteristics.
        
        Returns:
            Preferred level (1 or 2), or None for hierarchical
        """
        query_lower = query.lower()
        
        # Factual/definition queries → Fine-grained (L2)
        if any(kw in query_lower for kw in self.query_patterns['factual']):
            return 2  # Precise matching
        
        # Explanation/concept queries → Medium (L1)
        if any(kw in query_lower for kw in self.query_patterns['explanation']):
            return 1  # More context
        
        # Complex/multi-step → Use hierarchical (returns None)
        if any(kw in query_lower for kw in self.query_patterns['complex']):
            return None  # Trigger hierarchical
        
        # Default: Medium level (L1) - good balance
        return 1
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder."""
        
        logger.info(f"Reranking {len(results)} results")
        
        try:
            reranked = self.reranker.rerank(
                query,
                results,
                top_k=config.rerank_top_k,
                use_metadata_boost=config.use_metadata_boost
            )
            
            # Convert back to result format
            reranked_results = []
            for r in reranked:
                reranked_results.append({
                    'chunk_id': r.chunk_id,
                    'score': r.final_score,
                    'content': r.metadata,
                    'rerank_score': r.rerank_score,
                    'original_score': r.original_score
                })
            
            logger.info(f"✓ Reranked to top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:config.rerank_top_k]
    
    def _deduplicate_chunks(
        self, 
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Enhanced deduplication considering chunk similarity.
        
        Removes:
        1. Exact duplicates (same chunk_id)
        2. Near-duplicates (high text similarity)
        3. Parent-child pairs where child provides same info
        """
        
        if not chunks:
            return chunks
        
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            cid = chunk['chunk_id']
            
            # Skip exact duplicates
            if cid in seen_ids:
                continue
            
            # Check for near-duplicates
            is_duplicate = False
            chunk_text = chunk['content'].get('text', '').lower()
            
            for existing in unique_chunks:
                existing_text = existing['content'].get('text', '').lower()
                
                # Quick check: length difference
                len_ratio = min(len(chunk_text), len(existing_text)) / max(len(chunk_text), len(existing_text), 1)
                
                if len_ratio > similarity_threshold:
                    # Detailed check: token overlap
                    similarity = self._text_similarity(chunk_text, existing_text)
                    
                    if similarity > similarity_threshold:
                        # Near duplicate - keep higher scoring one
                        if chunk.get('score', 0) > existing.get('score', 0):
                            unique_chunks.remove(existing)
                            logger.debug(f"Replaced duplicate with higher-scoring chunk")
                        else:
                            is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
        
        if len(chunks) > len(unique_chunks):
            logger.info(f"Deduplicated: {len(chunks)} → {len(unique_chunks)} chunks")
        
        return unique_chunks
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Quick text similarity using token overlap (Jaccard)."""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _expand_context(
        self,
        results: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Expand results with parent or child chunks.
        Parent L0 chunks are non-embeddable but provide context.
        """
        
        expanded = list(results)
        chunk_ids = [r['chunk_id'] for r in results]
        
        if config.include_parents:
            parents = self.vector_store.get_parent_chunks(chunk_ids)
            # Mark as context-only
            for p in parents:
                p['content']['is_context_expansion'] = True
            expanded.extend(parents)
            logger.info(f"Added {len(parents)} parent chunks")
        
        if config.include_children:
            for chunk_id in chunk_ids:
                children = self.vector_store.get_children_chunks(chunk_id)
                for c in children:
                    c['content']['is_context_expansion'] = True
                expanded.extend(children)
            logger.info(f"Added children chunks")
        
        # Remove duplicates
        seen = set()
        unique = []
        for result in expanded:
            cid = result['chunk_id']
            if cid not in seen:
                seen.add(cid)
                unique.append(result)
        
        return unique
    
    def _assemble_context(
        self, 
        chunks: List[Dict[str, Any]],
        max_tokens: int = 4000
    ) -> str:
        """
        Enhanced context assembly with token-aware truncation.
        
        Features:
        - Smart sorting (level → document → position)
        - Token limit enforcement
        - Rich formatting with headers
        - Special markers for important content
        """
        
        if not chunks:
            return ""
        
        # Sort by level (lower first), then document position
        sorted_chunks = sorted(
            chunks,
            key=lambda x: (
                x['content'].get('level', 0),
                x['content'].get('doc_id', ''),
                x['content'].get('start_char', 0)
            )
        )
        
        context_parts = []
        current_tokens = 0
        chunks_included = 0
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk['content']
            text = content.get('text', '').strip()
            
            if not text:
                continue
            
            # Estimate tokens (rough: words * 1.3)
            chunk_tokens = int(len(text.split()) * 1.3)
            
            # Check if adding this chunk exceeds limit
            if current_tokens + chunk_tokens > max_tokens:
                print(f"Context truncated at {current_tokens} tokens (limit: {max_tokens})")
                logger.warning(f"Context truncated at {current_tokens} tokens (limit: {max_tokens})")
                continue
            
            # Add source header if new source or first chunk
            if i == 0 or content.get('source') != sorted_chunks[i-1]['content'].get('source'):
                header = self._format_chunk_header(content)
                context_parts.append(header)
            
            # Add formatted chunk text
            formatted_text = self._format_chunk_text(text, content)
            context_parts.append(formatted_text)
            context_parts.append("\n\n")
            
            current_tokens += chunk_tokens
            chunks_included += 1
        print(f"Assembled context: {chunks_included}/{len(chunks)} chunks, ~{current_tokens} tokens")
        logger.info(f"Assembled context: {chunks_included}/{len(chunks)} chunks, ~{current_tokens} tokens")
        
        return "".join(context_parts).strip()
    
    def _format_chunk_header(self, content: Dict) -> str:
        """Format chunk source header with rich information."""
        
        source = content.get('source', 'Unknown')
        grade = content.get('grade_level', 'General')
        difficulty = content.get('difficulty', 'Unknown')
        level = content.get('level', '?')
        embeddable = content.get('embeddable', True)
        
        # Determine chunk type
        if not embeddable:
            chunk_type = "📄 Context"
        else:
            chunk_type = f"📑 Level {level}"
        
        # Add special markers for important content
        markers = []
        if content.get('contains_theorem'):
            markers.append("📘 Theorem")
        if content.get('contains_formula'):
            markers.append("📐 Formula")
        if content.get('contains_shape'):
            markers.append("🔷 Shape")
        
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        
        return (
            f"\n{'─'*70}\n"
            f"📚 {source} | Grade {grade} | {difficulty} | {chunk_type}{marker_str}\n"
            f"{'─'*70}\n"
        )
    
    def _format_chunk_text(self, text: str, content: Dict) -> str:
        """Format chunk text with minimal processing."""
        # Clean excessive whitespace
        text = ' '.join(text.split())
        return text
    
    def _prepare_metadata(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        config: RetrievalConfig
    ) -> Dict[str, Any]:
        """Prepare comprehensive metadata about retrieval results."""
        
        if not chunks:
            return {
                'num_chunks': 0,
                'sources': [],
                'topics': [],
                'grade_levels': [],
                'difficulties': [],
                'embeddable_count': 0,
                'context_only_count': 0,
                'avg_score': 0.0
            }
        
        # Collect statistics
        sources = set()
        topics = set()
        grade_levels = set()
        difficulties = set()
        embeddable_count = 0
        context_only_count = 0
        level_distribution = {0: 0, 1: 0, 2: 0}
        
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
            
            # Count embeddable vs context-only
            if content.get('embeddable', True):
                embeddable_count += 1
            else:
                context_only_count += 1
            
            # Level distribution
            level = content.get('level', 0)
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Calculate average score
        avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks) if chunks else 0
        
        return {
            'num_chunks': len(chunks),
            'sources': list(sources),
            'topics': list(topics),
            'grade_levels': list(grade_levels),
            'difficulties': list(difficulties),
            'avg_score': avg_score,
            'strategy_used': config.strategy.value,
            'level_preference': config.level_preference,
            'reranked': config.rerank,
            'embeddable_count': embeddable_count,
            'context_only_count': context_only_count,
            'level_distribution': level_distribution,
            'embedding_limit': self.embedding_limit,
            'deduplication_enabled': config.enable_deduplication
        }
    
    # ========== Additional Methods ==========
    
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
        
        # Re-rank based on relevance feedback if provided
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
        
        # Boost scores of similar chunks
        for chunk in result.chunks:
            if chunk['chunk_id'] not in relevant_chunk_ids:
                # Apply similarity boost (simple version)
                chunk['score'] *= 1.1
        
        # Re-sort
        result.chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Reassemble context
        result.context = self._assemble_context(result.chunks, result.config.max_context_tokens)
        
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
        for i, (query, config) in enumerate(zip(queries, configs)):
            logger.info(f"Batch retrieval {i+1}/{len(queries)}: {query[:50]}...")
            result = self.retrieve(query, config)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        
        try:
            # Get index statistics
            stats = self.vector_store.es_client.indices.stats(
                index=self.vector_store.index_name
            )
            
            index_stats = stats['indices'][self.vector_store.index_name]
            total_count = index_stats['total']['docs']['count']
            
            # Query for embeddable chunks
            embeddable_query = {"query": {"term": {"embeddable": True}}}
            embeddable_count = self.vector_store.es_client.count(
                index=self.vector_store.index_name,
                body=embeddable_query
            )['count']
            
            # Get level distribution
            level_aggs = {
                "aggs": {
                    "levels": {
                        "terms": {"field": "level"}
                    }
                }
            }
            
            level_dist_response = self.vector_store.es_client.search(
                index=self.vector_store.index_name,
                body=level_aggs,
                size=0
            )
            
            level_distribution = {}
            for bucket in level_dist_response['aggregations']['levels']['buckets']:
                level_distribution[f"Level {bucket['key']}"] = bucket['doc_count']
            
            return {
                'status': 'healthy',
                'total_chunks': total_count,
                'embeddable_chunks': embeddable_count,
                'context_chunks': total_count - embeddable_count,
                'level_distribution': level_distribution,
                'index_size_mb': index_stats['total']['store']['size_in_bytes'] / (1024 * 1024),
                'embedding_model': getattr(settings, 'EMBEDDING_MODEL', 'unknown'),
                'embedding_limit': self.embedding_limit,
                'reranker_enabled': self.reranker is not None,
                'cache_enabled': True,
                'features': {
                    'hierarchical_retrieval': True,
                    'adaptive_strategy': True,
                    'level_aware_fusion': True,
                    'deduplication': True,
                    'token_aware_assembly': True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'embedding_limit': self.embedding_limit
            }