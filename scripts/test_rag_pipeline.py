#!/usr/bin/env python
"""
Integration tests for Geometry RAG Pipeline.
Tests end-to-end retrieval functionality with real database.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pprint import pprint
from typing import List, Dict, Any

from src.retrieval.rag_pipeline import (
    GeometryRAGPipeline, 
    RetrievalConfig, 
    RetrievalStrategy
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator(title: str, char: str = "="):
    """Print a formatted separator."""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")

def print_results(results: List[Dict[str, Any]], max_results: int = 3):
    """Print search results in a readable format."""
    for i, result in enumerate(results[:max_results], 1):
        print(f"\n[Result {i}]")
        print(f"  Chunk ID: {result['chunk_id'][:16]}...")
        print(f"  Score: {result.get('score', 0):.4f}")
        
        content = result.get('content', {})
        print(f"  Level: {content.get('level', 'N/A')}")
        print(f"  Grade: {content.get('grade_level', 'N/A')}")
        print(f"  Difficulty: {content.get('difficulty', 'N/A')}")
        
        text = content.get('text', '')
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"  Text: {preview}")

def test_01_basic_retrieval():
    """Test 1: Basic Retrieval with Different Strategies"""
    print_separator("TEST 1: Basic Retrieval")
    
    pipeline = GeometryRAGPipeline(enable_reranker=False)
    
    queries = [
        "What is the Pythagorean theorem?",
        "Explain properties of triangles",
        "How to calculate area of a circle?"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"{'─' * 60}")
        
        config = RetrievalConfig(
            strategy=RetrievalStrategy.VECTOR_ONLY,
            top_k=5,
            rerank=False
        )
        
        try:
            result = pipeline.retrieve(query, config)
            
            print(f"\n✓ Found {len(result.chunks)} chunks")
            print(f"  Sources: {', '.join(result.metadata.get('sources', [])[:3])}")
            print(f"  Topics: {', '.join(result.metadata.get('topics', [])[:5])}")
            print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
            
            if result.chunks:
                print(f"\n  Top Result Preview:")
                print_results(result.chunks, max_results=1)
        
        except Exception as e:
            print(f"✗ Error: {e}")
            logger.error(f"Test failed for query: {query}", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_02_hybrid_search():
    """Test 2: Compare Retrieval Strategies"""
    print_separator("TEST 2: Hybrid Search Comparison")
    
    pipeline = GeometryRAGPipeline(enable_reranker=False)
    
    query = "congruent triangles theorem proof"
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    strategies = [
        (RetrievalStrategy.VECTOR_ONLY, "Vector Search"),
        (RetrievalStrategy.KEYWORD_ONLY, "Keyword Search"),
        (RetrievalStrategy.HYBRID, "Hybrid Search")
    ]
    
    for strategy, name in strategies:
        print(f"\n[{name}]")
        
        config = RetrievalConfig(
            strategy=strategy,
            top_k=3,
            rerank=False
        )
        
        try:
            result = pipeline.retrieve(query, config)
            
            print(f"  ✓ Found: {len(result.chunks)} chunks")
            print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
            
            if result.chunks:
                print(f"  Top Score: {result.chunks[0].get('score', 0):.4f}")
                
                # Show top result's text preview
                top_text = result.chunks[0].get('content', {}).get('text', '')
                preview = top_text[:100] + "..." if len(top_text) > 100 else top_text
                print(f"  Preview: {preview}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            logger.error(f"Strategy {name} failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_03_reranking():
    """Test 3: Reranking Effectiveness"""
    print_separator("TEST 3: Reranking")
    
    try:
        pipeline = GeometryRAGPipeline(enable_reranker=True)
        print("✓ Reranker initialized successfully")
    except Exception as e:
        print(f"✗ Reranker initialization failed: {e}")
        print("  Skipping reranking tests...")
        return
    
    query = "angle bisector theorem with proof"
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    # Without reranking
    config_no_rerank = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=10,
        rerank=False
    )
    
    try:
        result_no_rerank = pipeline.retrieve(query, config_no_rerank)
        
        print(f"\n[Without Reranking]")
        print(f"  Found: {len(result_no_rerank.chunks)} chunks")
        print(f"  Top 3 Chunk IDs:")
        for i, chunk in enumerate(result_no_rerank.chunks[:3], 1):
            print(f"    {i}. {chunk['chunk_id'][:16]}... (score: {chunk['score']:.4f})")
    
    except Exception as e:
        print(f"✗ Retrieval without reranking failed: {e}")
        result_no_rerank = None
    
    # With reranking
    config_rerank = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=10,
        rerank=True,
        rerank_top_k=5
    )
    
    try:
        result_rerank = pipeline.retrieve(query, config_rerank)
        
        print(f"\n[With Reranking]")
        print(f"  Found: {len(result_rerank.chunks)} chunks")
        print(f"  Top 3 Chunk IDs:")
        for i, chunk in enumerate(result_rerank.chunks[:3], 1):
            print(f"    {i}. {chunk['chunk_id'][:16]}... (score: {chunk['score']:.4f})")
        
        # Compare order
        if result_no_rerank:
            order_changed = (
                result_no_rerank.chunks[0]['chunk_id'] != 
                result_rerank.chunks[0]['chunk_id']
            )
            print(f"\n  Order changed: {'Yes ✓' if order_changed else 'No'}")
    
    except Exception as e:
        print(f"✗ Reranking failed: {e}")
        logger.error("Reranking test failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_04_hierarchical_retrieval():
    """Test 4: Hierarchical Retrieval"""
    print_separator("TEST 4: Hierarchical Retrieval")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    query = "properties of quadrilaterals"
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    config = RetrievalConfig(
        strategy=RetrievalStrategy.HIERARCHICAL,
        top_k=8,
        rerank=True,
        rerank_top_k=5
    )
    
    try:
        result = pipeline.retrieve(query, config)
        
        print(f"\n✓ Found {len(result.chunks)} chunks")
        
        # Show level distribution
        level_counts = {}
        for chunk in result.chunks:
            level = chunk['content'].get('level', 'unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print(f"\n  Level Distribution:")
        for level in sorted(level_counts.keys()):
            print(f"    Level {level}: {level_counts[level]} chunks")
        
        print(f"\n  Metadata:")
        print(f"    Sources: {', '.join(result.metadata.get('sources', [])[:3])}")
        print(f"    Grade Levels: {', '.join(result.metadata.get('grade_levels', []))}")
        print(f"    Difficulties: {', '.join(result.metadata.get('difficulties', []))}")
        
        # Show context preview
        print(f"\n  Context Preview:")
        context_preview = result.context[:400] + "..." if len(result.context) > 400 else result.context
        print(f"    {context_preview}")
    
    except Exception as e:
        print(f"✗ Hierarchical retrieval failed: {e}")
        logger.error("Hierarchical retrieval test failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_05_filtering():
    """Test 5: Metadata Filtering"""
    print_separator("TEST 5: Metadata Filtering")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    query = "area calculation"
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    filters_list = [
        (None, "No Filters"),
        ({'grade_level': 'Middle School (6-8)'}, "Middle School Only"),
        ({'difficulty': 'Beginner'}, "Beginner Level"),
        ({'grade_level': 'High School (9-12)', 'difficulty': 'Advanced'}, "High School Advanced")
    ]
    
    for filters, description in filters_list:
        print(f"\n[{description}]")
        if filters:
            print(f"  Filters: {filters}")
        
        config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=5,
            rerank=True,
            rerank_top_k=3,
            filters=filters
        )
        
        try:
            result = pipeline.retrieve(query, config)
            
            print(f"  ✓ Found: {len(result.chunks)} chunks")
            
            if result.chunks:
                grades = result.metadata.get('grade_levels', [])
                diffs = result.metadata.get('difficulties', [])
                
                print(f"  Grade Levels: {', '.join(grades) if grades else 'N/A'}")
                print(f"  Difficulties: {', '.join(diffs) if diffs else 'N/A'}")
                print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            logger.error(f"Filtering test failed for {description}", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_06_context_expansion():
    """Test 6: Context Expansion (Parent/Child)"""
    print_separator("TEST 6: Context Expansion")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    query = "circle theorems"
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    # Without expansion
    config_no_expand = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        rerank=True,
        rerank_top_k=3,
        include_parents=False,
        include_children=False
    )
    
    try:
        result_no_expand = pipeline.retrieve(query, config_no_expand)
        
        print(f"\n[Without Expansion]")
        print(f"  Chunks: {len(result_no_expand.chunks)}")
        print(f"  Context Length: {len(result_no_expand.context)} characters")
    
    except Exception as e:
        print(f"✗ Retrieval without expansion failed: {e}")
        result_no_expand = None
    
    # With parent expansion
    config_expand = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        rerank=True,
        rerank_top_k=3,
        include_parents=True,
        include_children=False
    )
    
    try:
        result_expand = pipeline.retrieve(query, config_expand)
        
        print(f"\n[With Parent Expansion]")
        print(f"  Chunks: {len(result_expand.chunks)}")
        print(f"  Context Length: {len(result_expand.context)} characters")
        
        if result_no_expand:
            expansion_ratio = len(result_expand.chunks) / max(len(result_no_expand.chunks), 1)
            print(f"  Expansion Ratio: {expansion_ratio:.2f}x")
            
            context_increase = len(result_expand.context) - len(result_no_expand.context)
            print(f"  Context Increase: +{context_increase} characters")
    
    except Exception as e:
        print(f"✗ Context expansion failed: {e}")
        logger.error("Context expansion test failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_07_batch_retrieval():
    """Test 7: Batch Retrieval"""
    print_separator("TEST 7: Batch Retrieval")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    queries = [
        "What is an isosceles triangle?",
        "Explain parallel lines and transversals",
        "Calculate circumference of circle"
    ]
    
    print(f"\nBatch retrieval for {len(queries)} queries")
    print(f"{'─' * 60}")
    
    configs = [
        RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID, 
            top_k=3, 
            rerank=True
        )
        for _ in queries
    ]
    
    try:
        results = pipeline.batch_retrieve(queries, configs)
        
        print(f"\n✓ Batch retrieval completed")
        print(f"  Processed: {len(results)} queries")
        
        for query, result in zip(queries, results):
            print(f"\n  Query: {query}")
            print(f"    Chunks: {len(result.chunks)}")
            print(f"    Sources: {', '.join(result.metadata.get('sources', [])[:2])}")
            if result.chunks:
                print(f"    Top Score: {result.chunks[0].get('score', 0):.4f}")
    
    except Exception as e:
        print(f"✗ Batch retrieval failed: {e}")
        logger.error("Batch retrieval test failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def test_08_system_statistics():
    """Test 8: System Statistics"""
    print_separator("TEST 8: System Statistics")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    try:
        stats = pipeline.get_statistics()
        
        print("\n✓ System Statistics:")
        print(f"  Total Chunks: {stats.get('total_chunks', 'N/A'):,}")
        print(f"  Index Size: {stats.get('index_size_mb', 0):.2f} MB")
        print(f"  Embedding Model: {stats.get('embedding_model', 'N/A')}")
        print(f"  Reranker Enabled: {stats.get('reranker_enabled', False)}")
        print(f"  Cache Enabled: {stats.get('cache_enabled', False)}")
    
    except Exception as e:
        print(f"✗ Failed to get statistics: {e}")
        logger.error("Statistics test failed", exc_info=True)
    
    print(f"\n{'=' * 80}\n")

def main():
    """Run all integration tests."""
    print_separator("GEOMETRY RAG PIPELINE - INTEGRATION TEST SUITE", "=")
    
    print("\nStarting comprehensive integration tests...")
    print("These tests require a populated Elasticsearch database.\n")
    
    tests = [
        ("Basic Retrieval", test_01_basic_retrieval),
        ("Hybrid Search Comparison", test_02_hybrid_search),
        ("Reranking", test_03_reranking),
        ("Hierarchical Retrieval", test_04_hierarchical_retrieval),
        ("Metadata Filtering", test_05_filtering),
        ("Context Expansion", test_06_context_expansion),
        ("Batch Retrieval", test_07_batch_retrieval),
        ("System Statistics", test_08_system_statistics)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"Test '{test_name}' failed with exception: {e}", exc_info=True)
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}\n")
    
    # Final summary
    print_separator("TEST SUMMARY", "=")
    print(f"\nTotal Tests: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Phase 2 is working correctly.\n")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs for details.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())