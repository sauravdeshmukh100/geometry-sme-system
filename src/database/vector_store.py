# src/database/vector_store.py

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import numpy as np

from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import redis
import json
import hashlib

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeometryVectorStore:
    """Enhanced vector store with caching and advanced retrieval."""
    
    def __init__(self):
        self.es_client = Elasticsearch(
            hosts=[{
                "host": settings.ES_HOST,
                "port": settings.ES_PORT,
                "scheme": settings.ES_SCHEME
            }],
            request_timeout=settings.ES_TIMEOUT,
        )

        self.index_name = settings.ES_INDEX_NAME
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedder = SentenceTransformer(
            settings.embedding_model, 
            device=settings.device
        )
        
        # Initialize Redis cache
        self.cache = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False
        )
        self.cache_ttl = 3600  # 1 hour
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text with caching."""
        # Create cache key
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        
        # Generate embedding
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        
        # Cache the result
        self.cache.setex(
            cache_key,
            self.cache_ttl,
            embedding.tobytes()
        )
        
        return embedding
    
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        level: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build Elasticsearch query
        must_clauses = []
        
        # Add level filter if specified
        if level is not None:
            must_clauses.append({"term": {"level": level}})
        
        # Add custom filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})
        
        # Construct search body
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": must_clauses if must_clauses else [{"match_all": {}}]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding.tolist()
                        }
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]  # Don't return embeddings
            }
        }
        
        # Execute search
        response = self.es_client.search(
            index=self.index_name,
            body=search_body
        )
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            result = {
                'chunk_id': hit['_id'],
                'score': hit['_score'] - 1.0,  # Normalize score
                'content': hit['_source']
            }
            results.append(result)
        
        return results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        level: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        
        # Build must clauses
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "source", "topics"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]
        
        # Add level filter
        if level is not None:
            must_clauses.append({"term": {"level": level}})
        
        # Add custom filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})
        
        # Construct search body
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        # Execute search
        response = self.es_client.search(
            index=self.index_name,
            body=search_body
        )
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            result = {
                'chunk_id': hit['_id'],
                'score': hit['_score'],
                'content': hit['_source']
            }
            results.append(result)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        level: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search."""
        
        # Perform both searches
        vector_results = self.vector_search(
            query, 
            top_k=top_k * 2,  # Get more results for better fusion
            level=level,
            filters=filters
        )
        
        keyword_results = self.keyword_search(
            query,
            top_k=top_k * 2,
            level=level,
            filters=filters
        )
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in results:
                    r['normalized_score'] = (r['score'] - min_score) / score_range
            else:
                for r in results:
                    r['normalized_score'] = 1.0
            return results
        
        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Merge results using Reciprocal Rank Fusion
        chunk_scores = {}
        
        for rank, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    'content': result['content'],
                    'vector_score': 0,
                    'keyword_score': 0,
                    'vector_rank': None,
                    'keyword_rank': None
                }
            chunk_scores[chunk_id]['vector_score'] = result['normalized_score']
            chunk_scores[chunk_id]['vector_rank'] = rank + 1
        
        for rank, result in enumerate(keyword_results):
            chunk_id = result['chunk_id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    'content': result['content'],
                    'vector_score': 0,
                    'keyword_score': 0,
                    'vector_rank': None,
                    'keyword_rank': None
                }
            chunk_scores[chunk_id]['keyword_score'] = result['normalized_score']
            chunk_scores[chunk_id]['keyword_rank'] = rank + 1
        
        # Calculate hybrid scores
        k = 60  # RRF constant
        for chunk_id, data in chunk_scores.items():
            vector_rrf = 1.0 / (k + data['vector_rank']) if data['vector_rank'] else 0
            keyword_rrf = 1.0 / (k + data['keyword_rank']) if data['keyword_rank'] else 0
            
            data['hybrid_score'] = (
                vector_weight * vector_rrf + 
                keyword_weight * keyword_rrf
            )
        
        # Sort by hybrid score and return top_k
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )[:top_k]
        
        # Format results
        final_results = []
        for chunk_id, data in sorted_results:
            final_results.append({
                'chunk_id': chunk_id,
                'score': data['hybrid_score'],
                'vector_score': data['vector_score'],
                'keyword_score': data['keyword_score'],
                'content': data['content']
            })
        
        return final_results
    
    def get_parent_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve parent chunks for given chunk IDs."""
        
        # Get chunks
        chunks = self.es_client.mget(
            index=self.index_name,
            body={"ids": chunk_ids}
        )
        
        parent_ids = set()
        for doc in chunks['docs']:
            if doc['found'] and doc['_source'].get('parent_id'):
                parent_ids.add(doc['_source']['parent_id'])
        
        if not parent_ids:
            return []
        
        # Get parent chunks
        parents = self.es_client.mget(
            index=self.index_name,
            body={"ids": list(parent_ids)}
        )
        
        results = []
        for doc in parents['docs']:
            if doc['found']:
                results.append({
                    'chunk_id': doc['_id'],
                    'content': doc['_source']
                })
        
        return results
    
    def get_children_chunks(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Retrieve children chunks for a given chunk."""
        
        search_body = {
            "query": {
                "term": {"parent_id": chunk_id}
            },
            "size": 100,
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        response = self.es_client.search(
            index=self.index_name,
            body=search_body
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'chunk_id': hit['_id'],
                'content': hit['_source']
            })
        
        return results
    
    def clear_cache(self):
        """Clear the Redis cache."""
        self.cache.flushdb()
        logger.info("Cache cleared")