# tests/test_reranker.py

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.reranker import GeometryReranker, GeometryMetadataScorer

@pytest.fixture
def reranker():
    """Create reranker instance."""
    return GeometryReranker()

@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    return [
        {
            'chunk_id': 'chunk1',
            'score': 0.8,
            'content': {
                'text': 'The Pythagorean theorem states that a² + b² = c²',
                'contains_theorem': True,
                'contains_formula': True,
                'topic_density': 0.15
            }
        },
        {
            'chunk_id': 'chunk2',
            'score': 0.75,
            'content': {
                'text': 'A triangle has three sides and three angles.',
                'contains_theorem': False,
                'contains_formula': False,
                'topic_density': 0.08
            }
        }
    ]

def test_reranker_initialization(reranker):
    """Test reranker initializes correctly."""
    assert reranker.model is not None
    assert reranker.model_name == "BAAI/bge-reranker-base"

def test_rerank_basic(reranker, sample_results):
    """Test basic reranking."""
    query = "Pythagorean theorem proof"
    reranked = reranker.rerank(query, sample_results, top_k=2)
    
    assert len(reranked) <= 2
    assert all(hasattr(r, 'final_score') for r in reranked)
    assert all(hasattr(r, 'rerank_score') for r in reranked)

def test_metadata_boost(reranker, sample_results):
    """Test metadata boosting."""
    query = "theorem"
    
    # With metadata boost
    with_boost = reranker.rerank(
        query, 
        sample_results, 
        use_metadata_boost=True
    )
    
    # Without metadata boost
    without_boost = reranker.rerank(
        query, 
        sample_results, 
        use_metadata_boost=False
    )
    
    # Scores should be different
    assert with_boost[0].final_score != without_boost[0].final_score
    
def test_metadata_scorer():
    """Test metadata scorer."""
    scorer = GeometryMetadataScorer()
    
    query_context = {
        'grade_level': 'Middle School (6-8)',
        'difficulty': 'Intermediate',
        'topics': ['theorem', 'triangle']
    }
    
    results = [
        {
            'content': {
                'grade_level': 'Middle School (6-8)',
                'difficulty': 'Intermediate',
                'topics': ['theorem', 'triangle', 'proof']
            }
        }
    ]
    
    scores = scorer.score_by_metadata(query_context, results)
    
    assert len(scores) == 1
    assert scores[0] > 0  # Should have positive score for matches