# tests/test_rag_pipeline.py

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.rag_pipeline import (
    GeometryRAGPipeline,
    RetrievalConfig,
    RetrievalStrategy
)

@pytest.fixture
def pipeline():
    """Create RAG pipeline instance."""
    return GeometryRAGPipeline(enable_reranker=False)  # Disable for faster tests

def test_pipeline_initialization(pipeline):
    """Test pipeline initializes correctly."""
    assert pipeline.vector_store is not None
    assert pipeline.metadata_scorer is not None

def test_basic_retrieval(pipeline):
    """Test basic retrieval."""
    query = "triangle properties"
    config = RetrievalConfig(
        strategy=RetrievalStrategy.VECTOR_ONLY,
        top_k=5,
        rerank=False
    )
    
    result = pipeline.retrieve(query, config)
    
    assert result.query == query
    assert isinstance(result.chunks, list)
    assert isinstance(result.context, str)
    assert isinstance(result.metadata, dict)

def test_hybrid_retrieval(pipeline):
    """Test hybrid retrieval."""
    query = "Pythagorean theorem"
    config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=5
    )
    
    result = pipeline.retrieve(query, config)
    
    assert len(result.chunks) <= 5
    assert 'strategy_used' in result.metadata
    assert result.metadata['strategy_used'] == 'hybrid'

def test_filtering(pipeline):
    """Test retrieval with filters."""
    query = "area calculation"
    config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=5,
        filters={'grade_level': 'Middle School (6-8)'}
    )
    
    result = pipeline.retrieve(query, config)
    
    # Check that results respect filter (if data exists)
    if result.chunks:
        assert 'grade_levels' in result.metadata

def test_batch_retrieval(pipeline):
    """Test batch retrieval."""
    queries = ["triangle", "circle", "angle"]
    
    results = pipeline.batch_retrieve(queries)
    
    assert len(results) == len(queries)
    assert all(isinstance(r.chunks, list) for r in results)

def test_context_assembly(pipeline):
    """Test context assembly."""
    query = "geometry basics"
    config = RetrievalConfig(top_k=3)
    
    result = pipeline.retrieve(query, config)
    
    # Context should be non-empty if chunks exist
    if result.chunks:
        assert len(result.context) > 0
        assert isinstance(result.context, str)

def test_statistics(pipeline):
    """Test system statistics."""
    stats = pipeline.get_statistics()
    
    assert 'total_chunks' in stats
    assert 'embedding_model' in stats
    assert isinstance(stats['total_chunks'], int)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])