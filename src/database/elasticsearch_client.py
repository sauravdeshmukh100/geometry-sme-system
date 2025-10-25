from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.helpers import bulk, async_bulk
import numpy as np

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeometryElasticsearchClient:
    """Elasticsearch client for geometry content indexing and retrieval."""
    
    def __init__(self):
        es_url = f"http://{settings.ES_HOST}:{settings.ES_PORT}"

        self.client = Elasticsearch(
            [es_url],
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=settings.ES_TIMEOUT,
            retry_on_timeout=True,
        )
        self.async_client = AsyncElasticsearch(
            [es_url],
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=settings.ES_TIMEOUT,
            retry_on_timeout=True,
        )

        self.index_name = settings.ES_INDEX_NAME
        
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
                        
                        # Source information
                        "source": {"type": "keyword"},
                        "source_file": {"type": "keyword"},
                        
                        # Position in document
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                        
                        # Embeddings
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

    def bulk_index(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Bulk index documents to Elasticsearch."""
        from elasticsearch.helpers import bulk
        
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_id": doc['chunk_id'],
                "_source": {
                    **doc,
                    "indexed_at": datetime.now().isoformat()
                }
            }
            actions.append(action)
        
        # Index in batches
        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]
            success, failed = bulk(self.client, batch, raise_on_error=False)
            logger.info(f"Indexed batch: {success} successful, {failed} failed")
            
        self.client.indices.refresh(index=self.index_name)        