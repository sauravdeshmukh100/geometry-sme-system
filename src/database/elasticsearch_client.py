from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeometryElasticsearchClient:
    """
    Elasticsearch client for geometry content indexing and retrieval.
    Updated to handle chunks with and without embeddings (Level 0 context chunks).
    """
    
    def __init__(self):
        self.client = Elasticsearch(
            [{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'http'}],
            timeout=settings.es_timeout
        )
        self.index_name = settings.es_index_name
        
    def create_index(self, recreate: bool = False):
        """Create Elasticsearch index with appropriate mappings."""
        
        if recreate and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")
        
        if not self.client.indices.exists(index=self.index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        # Chunk identifiers
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        
                        # Content
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "search_analyzer": "standard"
                        },
                        
                        # Hierarchical level
                        "level": {"type": "integer"},
                        
                        # NEW: Embeddability flag
                        "embeddable": {"type": "boolean"},
                        
                        # Source information
                        "source": {"type": "keyword"},
                        "source_file": {"type": "keyword"},
                        "source_type": {"type": "keyword"},
                        
                        # Position in document
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                        
                        # Embeddings (OPTIONAL - Level 0 chunks don't have this)
                        "embedding": {
                            "type": "dense_vector",
                            "dims": settings.embedding_dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # Geometry-specific fields
                        "grade_level": {"type": "keyword"},
                        "difficulty": {"type": "keyword"},
                        "topics": {"type": "keyword"},
                        "contains_theorem": {"type": "boolean"},
                        "contains_formula": {"type": "boolean"},
                        "contains_shape": {"type": "boolean"},
                        "contains_angle": {"type": "boolean"},
                        "topic_density": {"type": "float"},
                        
                        # Chapter info
                        "chapter_info": {"type": "object", "enabled": False},
                        
                        # Metadata
                        "metadata": {"type": "object", "enabled": False},
                        "indexed_at": {"type": "date"},
                        
                        # For BM25 search
                        "text_keywords": {
                            "type": "keyword",
                            "normalizer": "lowercase_normalizer"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "normalizer": {
                            "lowercase_normalizer": {
                                "type": "custom",
                                "filter": ["lowercase", "asciifolding"]
                            }
                        }
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=mappings)
            logger.info(f"Created index: {self.index_name}")
    
    def bulk_index(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Bulk index documents to Elasticsearch.
        Handles both embeddable chunks (with embeddings) and context chunks (without).
        """
        actions = []
        skipped_embeddings = 0
        
        for doc in documents:
            # Prepare document for indexing
            source_doc = {
                "chunk_id": doc['chunk_id'],
                "doc_id": doc['doc_id'],
                "parent_id": doc.get('parent_id'),
                "text": doc['text'],
                "level": doc['level'],
                "embeddable": doc.get('embeddable', True),
                "source": doc['source'],
                "source_type": doc.get('source_type', 'Unknown'),
                "start_char": doc['start_char'],
                "end_char": doc['end_char'],
                "grade_level": doc.get('grade_level', 'Unknown'),
                "difficulty": doc.get('difficulty', 'Unknown'),
                "topics": doc.get('topics', []),
                "contains_theorem": doc.get('contains_theorem', False),
                "contains_formula": doc.get('contains_formula', False),
                "contains_shape": doc.get('contains_shape', False),
                "contains_angle": doc.get('contains_angle', False),
                "topic_density": doc.get('topic_density', 0.0),
                "chapter_info": doc.get('chapter_info'),
                "metadata": doc.get('metadata', {}),
                "indexed_at": datetime.now().isoformat()
            }
            
            # Add embedding ONLY if present (Level 1 & 2 chunks)
            if doc.get('embedding') is not None:
                source_doc['embedding'] = doc['embedding']
            else:
                # Level 0 chunks don't have embeddings
                skipped_embeddings += 1
            
            action = {
                "_index": self.index_name,
                "_id": doc['chunk_id'],
                "_source": source_doc
            }
            actions.append(action)
        
        logger.info(f"Indexing {len(actions)} documents "
                   f"({len(actions) - skipped_embeddings} with embeddings, "
                   f"{skipped_embeddings} context-only)")
        
        # Index in batches
        success_count = 0
        error_count = 0
        
        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]
            try:
                success, failed = bulk(self.client, batch, raise_on_error=False)
                success_count += success
                
                if failed:
                    error_count += len(failed)
                    for item in failed:
                        logger.error(f"Failed to index: {item}")
                
                logger.info(f"Batch {i//batch_size + 1}: "
                           f"{success} successful, {len(failed) if failed else 0} failed")
            except Exception as e:
                logger.error(f"Batch indexing error: {e}")
                error_count += len(batch)
        
        # Refresh index
        self.client.indices.refresh(index=self.index_name)
        
        logger.info(f"Indexing complete: {success_count} successful, {error_count} failed")
        
        return success_count, error_count
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single chunk by ID."""
        try:
            result = self.client.get(index=self.index_name, id=chunk_id)
            return result['_source']
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search chunks using BM25 text search."""
        search_body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "source", "topics"],
                    "type": "best_fields"
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            
            # Count embeddable vs context-only chunks
            embeddable_query = {
                "query": {"term": {"embeddable": True}}
            }
            embeddable_count = self.client.count(
                index=self.index_name,
                body=embeddable_query
            )['count']
            
            total_count = index_stats['total']['docs']['count']
            
            return {
                'total_chunks': total_count,
                'embeddable_chunks': embeddable_count,
                'context_chunks': total_count - embeddable_count,
                'index_size_mb': index_stats['total']['store']['size_in_bytes'] / (1024 * 1024),
                'index_name': self.index_name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}