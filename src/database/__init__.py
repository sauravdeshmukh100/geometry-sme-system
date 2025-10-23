# src/database/__init__.py (UPDATED)

"""
Database module for Geometry SME system.
Handles Elasticsearch indexing and vector storage.
"""

from .elasticsearch_client import GeometryElasticsearchClient
from .vector_store import GeometryVectorStore

__all__ = [
    'GeometryElasticsearchClient',
    'GeometryVectorStore'
]

__version__ = '0.2.0'
