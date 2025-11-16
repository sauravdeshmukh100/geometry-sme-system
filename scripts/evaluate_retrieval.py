#!/usr/bin/env python
"""
Comprehensive Retrieval Evaluation Script
Tests retrieval quality, strategy comparison, and performance metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass
import json

from src.retrieval.rag_pipeline import (
    GeometryRAGPipeline, 
    RetrievalConfig, 
    RetrievalStrategy,
    RetrievalResult
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """Test query with expected outcomes."""
    query: str
    expected_concepts: List[str]
    expected_grade: str
    query_type: str  # factual, explanation, complex, listing
    description: str


# Comprehensive test queries covering different scenarios
TEST_QUERIES = [
    # Factual queries (expect Level 2, precise chunks)
    TestQuery(
        query="What is the Pythagorean theorem?",
        expected_concepts=['pythagorean', 'theorem', 'right triangle', 'hypotenuse', 'a²+b²=c²'],
        expected_grade="Grade 9",
        query_type="factual",
        description="Basic theorem definition"
    ),
    TestQuery(
        query="Define an isosceles triangle",
        expected_concepts=['isosceles', 'two equal sides', 'base angles', 'equal'],
        expected_grade="Grade 7",
        query_type="factual",
        description="Shape definition"
    ),
    TestQuery(
        query="What is the formula for the area of a circle?",
        expected_concepts=['circle', 'area', 'πr²', 'radius', 'formula'],
        expected_grade="Grade 8",
        query_type="factual",
        description="Formula query"
    ),
    
    # Explanation queries (expect Level 1, more context)
    TestQuery(
        query="Explain the properties of parallel lines",
        expected_concepts=['parallel', 'lines', 'never meet', 'equidistant', 'transversal'],
        expected_grade="Grade 7",
        query_type="explanation",
        description="Concept explanation"
    ),
    TestQuery(
        query="How does the angle sum property of triangles work?",
        expected_concepts=['triangle', 'angle', 'sum', '180', 'degrees'],
        expected_grade="Grade 7",
        query_type="explanation",
        description="Property explanation"
    ),
    TestQuery(
        query="Describe the types of quadrilaterals",
        expected_concepts=['quadrilateral', 'square', 'rectangle', 'parallelogram', 'trapezoid'],
        expected_grade="Grade 8",
        query_type="listing",
        description="Classification query"
    ),
    
    # Complex queries (expect hierarchical, multi-level)
    TestQuery(
        query="Prove that the sum of angles in a triangle is 180 degrees",
        expected_concepts=['triangle', 'angles', 'sum', '180', 'proof', 'parallel'],
        expected_grade="Grade 9",
        query_type="complex",
        description="Proof query"
    ),
    TestQuery(
        query="Calculate the area of a triangle given base and height",
        expected_concepts=['triangle', 'area', 'base', 'height', 'formula', '½bh'],
        expected_grade="Grade 8",
        query_type="complex",
        description="Calculation query"
    ),
    
    # Grade-specific queries
    TestQuery(
        query="Properties of circles for Grade 6",
        expected_concepts=['circle', 'radius', 'diameter', 'circumference'],
        expected_grade="Grade 6",
        query_type="explanation",
        description="Grade-specific content"
    ),
    TestQuery(
        query="Advanced theorems about triangles for Grade 10",
        expected_concepts=['triangle', 'theorem', 'congruence', 'similarity'],
        expected_grade="Grade 10",
        query_type="complex",
        description="Advanced content"
    ),
]


def print_separator(title: str, char: str = "="):
    """Print formatted separator."""
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")


def calculate_relevance_score(
    result: RetrievalResult,
    test_query: TestQuery
) -> Dict[str, float]:
    """
    Calculate comprehensive relevance score.
    
    Returns:
        Dictionary with individual scores and total
    """
    
    if not result.chunks:
        return {
            'concept_coverage': 0.0,
            'grade_match': 0.0,
            'retrieval_quality': 0.0,
            'level_appropriateness': 0.0,
            'total': 0.0
        }
    
    context_lower = result.context.lower()
    
    # 1. Concept Coverage (40% weight)
    concepts_found = sum(
        1 for concept in test_query.expected_concepts 
        if concept.lower() in context_lower
    )
    concept_score = (concepts_found / len(test_query.expected_concepts)) * 40
    
    # 2. Grade Level Match (20% weight)
    grade_match = any(
        chunk['content'].get('grade_level') == test_query.expected_grade
        for chunk in result.chunks
    )
    grade_score = 20 if grade_match else 0
    
    # 3. Retrieval Quality - based on scores (20% weight)
    avg_retrieval_score = result.metadata.get('avg_score', 0)
    retrieval_score = min(avg_retrieval_score, 1.0) * 20
    
    # 4. Level Appropriateness (20% weight)
    # Check if correct levels were retrieved based on query type
    level_dist = result.metadata.get('level_distribution', {})
    
    if test_query.query_type == 'factual':
        # Should prefer Level 2 (fine-grained)
        l2_count = level_dist.get(2, 0)
        level_score = min(l2_count / len(result.chunks), 1.0) * 20
    elif test_query.query_type == 'explanation':
        # Should have mix of Level 1 and 2
        l1_count = level_dist.get(1, 0)
        level_score = min(l1_count / len(result.chunks), 1.0) * 20
    elif test_query.query_type == 'complex':
        # Should use hierarchical (multiple levels)
        levels_used = sum(1 for v in level_dist.values() if v > 0)
        level_score = min(levels_used / 2, 1.0) * 20  # Expect at least 2 levels
    else:
        level_score = 10  # Neutral
    
    total_score = concept_score + grade_score + retrieval_score + level_score
    
    return {
        'concept_coverage': concept_score,
        'grade_match': grade_score,
        'retrieval_quality': retrieval_score,
        'level_appropriateness': level_score,
        'total': total_score
    }


def test_single_query(
    pipeline: GeometryRAGPipeline,
    test_query: TestQuery,
    config: RetrievalConfig,
    verbose: bool = False
) -> Dict[str, Any]:
    """Test a single query and return results."""
    
    start_time = time.time()
    
    try:
        result = pipeline.retrieve(test_query.query, config)
        elapsed_time = time.time() - start_time
        
        # Calculate scores
        scores = calculate_relevance_score(result, test_query)
        
        # Prepare test result
        test_result = {
            'query': test_query.query,
            'description': test_query.description,
            'query_type': test_query.query_type,
            'scores': scores,
            'metadata': {
                'num_chunks': len(result.chunks),
                'strategy_used': result.metadata.get('strategy_used'),
                'level_distribution': result.metadata.get('level_distribution', {}),
                'sources': result.metadata.get('sources', []),
                'embeddable_count': result.metadata.get('embeddable_count', 0),
                'context_only_count': result.metadata.get('context_only_count', 0),
                'elapsed_time': elapsed_time
            },
            'success': True
        }
        
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Query: {test_query.query}")
            print(f"Type: {test_query.query_type}")
            print(f"{'─'*70}")
            print(f"\nScores:")
            for metric, score in scores.items():
                print(f"  {metric:25s}: {score:5.1f}")
            print(f"\nMetadata:")
            print(f"  Chunks: {test_result['metadata']['num_chunks']}")
            print(f"  Strategy: {test_result['metadata']['strategy_used']}")
            print(f"  Levels: {test_result['metadata']['level_distribution']}")
            print(f"  Time: {elapsed_time:.2f}s")
        
        return test_result
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return {
            'query': test_query.query,
            'success': False,
            'error': str(e)
        }


def compare_strategies():
    """Compare different retrieval strategies."""
    
    print_separator("STRATEGY COMPARISON TEST", "=")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    strategies = [
        (RetrievalStrategy.HYBRID, "Hybrid (Vector + Keyword)"),
        (RetrievalStrategy.HIERARCHICAL, "Hierarchical (Multi-level)"),
        (RetrievalStrategy.ADAPTIVE, "Adaptive (Auto-select)"),
        (RetrievalStrategy.VECTOR_ONLY, "Vector Only"),
    ]
    
    strategy_results = {}
    
    for strategy, description in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {description}")
        print(f"{'='*70}")
        
        config = RetrievalConfig(
            strategy=strategy,
            top_k=10,
            rerank=True,
            rerank_top_k=5
        )
        
        query_results = []
        
        for test_query in TEST_QUERIES:
            result = test_single_query(pipeline, test_query, config, verbose=False)
            if result.get('success'):
                query_results.append(result)
            
            # Print progress
            score = result.get('scores', {}).get('total', 0)
            print(f"  {test_query.query_type:12s}: {score:5.1f} - {test_query.query[:40]}...")
        
        # Calculate average scores
        if query_results:
            avg_scores = {
                'concept_coverage': sum(r['scores']['concept_coverage'] for r in query_results) / len(query_results),
                'grade_match': sum(r['scores']['grade_match'] for r in query_results) / len(query_results),
                'retrieval_quality': sum(r['scores']['retrieval_quality'] for r in query_results) / len(query_results),
                'level_appropriateness': sum(r['scores']['level_appropriateness'] for r in query_results) / len(query_results),
                'total': sum(r['scores']['total'] for r in query_results) / len(query_results)
            }
            
            avg_time = sum(r['metadata']['elapsed_time'] for r in query_results) / len(query_results)
            
            strategy_results[description] = {
                'scores': avg_scores,
                'avg_time': avg_time,
                'num_queries': len(query_results)
            }
            
            print(f"\n  Average Total Score: {avg_scores['total']:.1f}/100")
            print(f"  Average Time: {avg_time:.2f}s")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    
    # Sort by total score
    sorted_strategies = sorted(
        strategy_results.items(),
        key=lambda x: x[1]['scores']['total'],
        reverse=True
    )
    
    print(f"\n{'Strategy':<30s} {'Total':<10s} {'Concept':<10s} {'Grade':<10s} {'Level':<10s} {'Time':<10s}")
    print("─"*70)
    
    for strategy_name, data in sorted_strategies:
        scores = data['scores']
        print(f"{strategy_name:<30s} "
              f"{scores['total']:>6.1f}    "
              f"{scores['concept_coverage']:>6.1f}    "
              f"{scores['grade_match']:>6.1f}    "
              f"{scores['level_appropriateness']:>6.1f}    "
              f"{data['avg_time']:>6.2f}s")
    
    return strategy_results


def test_query_types():
    """Test performance on different query types."""
    
    print_separator("QUERY TYPE PERFORMANCE TEST", "=")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    # Use adaptive strategy
    config = RetrievalConfig(
        strategy=RetrievalStrategy.ADAPTIVE,
        top_k=10,
        rerank=True,
        rerank_top_k=5
    )
    
    # Group queries by type
    query_types = {}
    for test_query in TEST_QUERIES:
        qtype = test_query.query_type
        if qtype not in query_types:
            query_types[qtype] = []
        query_types[qtype].append(test_query)
    
    # Test each type
    type_results = {}
    
    for qtype, queries in query_types.items():
        print(f"\n{'─'*70}")
        print(f"Query Type: {qtype.upper()}")
        print(f"{'─'*70}")
        
        results = []
        
        for test_query in queries:
            result = test_single_query(pipeline, test_query, config, verbose=True)
            if result.get('success'):
                results.append(result)
        
        if results:
            avg_score = sum(r['scores']['total'] for r in results) / len(results)
            type_results[qtype] = {
                'avg_score': avg_score,
                'num_queries': len(results)
            }
    
    # Summary
    print("\n" + "="*70)
    print("QUERY TYPE SUMMARY")
    print("="*70)
    
    for qtype, data in sorted(type_results.items(), key=lambda x: x[1]['avg_score'], reverse=True):
        print(f"{qtype:<15s}: {data['avg_score']:5.1f}/100 ({data['num_queries']} queries)")


def test_level_selection():
    """Test automatic level selection."""
    
    print_separator("LEVEL SELECTION TEST", "=")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    # Test queries that should select different levels
    test_cases = [
        ("What is a triangle?", 2, "Factual → Level 2"),
        ("Explain the Pythagorean theorem", 1, "Explanation → Level 1"),
        ("Prove triangle similarity", None, "Complex → Hierarchical"),
    ]
    
    for query, expected_level, description in test_cases:
        config = RetrievalConfig(
            strategy=RetrievalStrategy.ADAPTIVE,
            top_k=5
        )
        
        result = pipeline.retrieve(query, config)
        
        actual_strategy = result.metadata.get('strategy_used')
        level_dist = result.metadata.get('level_distribution', {})
        
        print(f"\n{'─'*70}")
        print(f"Query: {query}")
        print(f"Expected: {description}")
        print(f"{'─'*70}")
        print(f"Strategy Used: {actual_strategy}")
        print(f"Level Distribution: {level_dist}")
        print(f"Score: {result.metadata.get('avg_score', 0):.4f}")


def test_performance_metrics():
    """Test performance and efficiency."""
    
    print_separator("PERFORMANCE METRICS TEST", "=")
    
    pipeline = GeometryRAGPipeline(enable_reranker=True)
    
    # Test different configurations
    configs = [
        ("Default", RetrievalConfig()),
        ("With Rerank", RetrievalConfig(rerank=True)),
        ("Without Rerank", RetrievalConfig(rerank=False)),
        ("Hierarchical", RetrievalConfig(strategy=RetrievalStrategy.HIERARCHICAL)),
        ("Top-K=5", RetrievalConfig(top_k=5)),
        ("Top-K=20", RetrievalConfig(top_k=20)),
    ]
    
    # Use a subset of queries
    test_queries = TEST_QUERIES[:5]
    
    performance_results = {}
    
    for config_name, config in configs:
        print(f"\nTesting: {config_name}")
        
        times = []
        
        for test_query in test_queries:
            start = time.time()
            result = pipeline.retrieve(test_query.query, config)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        performance_results[config_name] = avg_time
        
        print(f"  Average time: {avg_time:.3f}s")
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    for config_name, avg_time in sorted(performance_results.items(), key=lambda x: x[1]):
        print(f"{config_name:<25s}: {avg_time:.3f}s")


def export_results(results: Dict[str, Any], filename: str = "evaluation_results.json"):
    """Export evaluation results to JSON."""
    
    output_path = os.path.join("scripts", "logs", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results exported to: {output_path}")


def main():
    """Run comprehensive evaluation."""
    
    print("="*70)
    print("  GEOMETRY SME - RETRIEVAL EVALUATION")
    print("  Comprehensive Testing Suite")
    print("="*70)
    
    print("\nAvailable Tests:")
    print("1. Strategy Comparison")
    print("2. Query Type Performance")
    print("3. Level Selection")
    print("4. Performance Metrics")
    print("5. All Tests")
    print("0. Exit")
    
    choice = input("\nSelect test (0-5): ").strip()
    
    try:
        if choice == '1':
            results = compare_strategies()
            export_results(results, "strategy_comparison.json")
        
        elif choice == '2':
            test_query_types()
        
        elif choice == '3':
            test_level_selection()
        
        elif choice == '4':
            test_performance_metrics()
        
        elif choice == '5':
            print("\n" + "="*70)
            print("  RUNNING ALL TESTS")
            print("="*70)
            
            results = compare_strategies()
            test_query_types()
            test_level_selection()
            test_performance_metrics()
            
            export_results(results, "complete_evaluation.json")
        
        elif choice == '0':
            print("\nGoodbye!")
            return 0
        
        else:
            print("\n❌ Invalid choice")
            return 1
        
        print("\n" + "="*70)
        print("  ✓ EVALUATION COMPLETE")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        logger.error("Evaluation failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())