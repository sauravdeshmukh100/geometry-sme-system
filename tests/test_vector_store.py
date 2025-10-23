# tests/test_vector_store.py

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.vector_store import GeometryVectorStore
from src.config.settings import settings

@pytest.fixture
def vector_store():
    """Create vector store instance for testing."""
    return GeometryVectorStore()

def test_vector_store_initialization(vector_store):
    """Test vector store initializes correctly."""
    assert vector_store.embedder is not None
    assert vector_store.es_client is not None
    assert vector_store.cache is not None

def test_generate_embedding(vector_store):
    """Test embedding generation."""
    text = "What is a triangle?"
    embedding = vector_store.generate_embedding(text)
    
    assert embedding is not None
    assert len(embedding) == settings.embedding_dimension
    assert embedding.dtype == 'float32'

def test_vector_search(vector_store):
    """Test vector search functionality."""
    query = "properties of triangles"
    results = vector_store.vector_search(query, top_k=5)
    
    assert isinstance(results, list)
    if results:  # If database has data
        assert 'chunk_id' in results[0]
        assert 'score' in results[0]
        assert 'content' in results[0]

def test_keyword_search(vector_store):
    """Test keyword search functionality."""
    query = "Pythagorean theorem"
    results = vector_store.keyword_search(query, top_k=5)
    
    assert isinstance(results, list)
    if results:
        assert 'chunk_id' in results[0]
        assert 'score' in results[0]

def test_hybrid_search(vector_store):
    """Test hybrid search."""
    query = "circle area formula"
    results = vector_store.hybrid_search(query, top_k=5)
    
    assert isinstance(results, list)
    if results:
        assert 'hybrid_score' in results[0]
        assert 'vector_score' in results[0]
        assert 'keyword_score' in results[0]

def test_parent_chunk_retrieval(vector_store):
    """Test parent chunk retrieval."""
    # This test requires actual data in the database
    # Get some chunk IDs first
    results = vector_store.vector_search("triangle", top_k=3)
    
    if results:
        chunk_ids = [r['chunk_id'] for r in results]
        parents = vector_store.get_parent_chunks(chunk_ids)
        
        assert isinstance(parents, list)
        # Parents may or may not exist, so we just check type

def test_caching(vector_store):
    """Test embedding caching."""
    text = "test caching"
    
    # Clear cache first
    vector_store.clear_cache()
    
    # First call - not cached
    embedding1 = vector_store.generate_embedding(text)
    
    # Second call - should be cached
    embedding2 = vector_store.generate_embedding(text)
    
    # Should be identical
    assert (embedding1 == embedding2).all()