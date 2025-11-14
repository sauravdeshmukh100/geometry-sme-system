#!/usr/bin/env python
"""Analyze pipeline logs for insights and patterns"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any
import statistics

def load_logs(log_dir: str = "logs/pipeline_traces") -> List[Dict[str, Any]]:
    """Load all log files from directory."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return []
    
    logs = []
    for log_file in log_path.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                logs.append(json.load(f))
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return logs

def analyze_session_types(logs: List[Dict]) -> Dict[str, int]:
    """Count sessions by type."""
    return Counter(log['session_type'] for log in logs)

def analyze_retrieval_performance(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze retrieval stage performance."""
    retrieval_data = []
    
    for log in logs:
        stages = log.get('stages', {})
        if '3_retrieval' in stages:
            data = stages['3_retrieval']['data']
            retrieval_data.append({
                'num_chunks': data.get('num_chunks_retrieved', 0),
                'context_length': data.get('context_length', 0),
                'avg_score': data.get('retrieval_metadata', {}).get('avg_score', 0),
                'strategy': data.get('config', {}).get('strategy', 'unknown')
            })
    
    if not retrieval_data:
        return {}
    
    return {
        'total_retrievals': len(retrieval_data),
        'avg_chunks': statistics.mean(r['num_chunks'] for r in retrieval_data),
        'avg_context_length': statistics.mean(r['context_length'] for r in retrieval_data),
        'avg_score': statistics.mean(r['avg_score'] for r in retrieval_data if r['avg_score'] > 0),
        'strategies_used': Counter(r['strategy'] for r in retrieval_data)
    }

def analyze_llm_performance(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze LLM generation performance."""
    llm_data = []
    
    for log in logs:
        stages = log.get('stages', {})
        if '5_llm_output' in stages:
            data = stages['5_llm_output']['data']
            llm_data.append({
                'generation_time': data.get('generation_time_seconds', 0),
                'response_length': data.get('response_length', 0),
                'model': data.get('model_info', {}).get('model', 'unknown')
            })
    
    if not llm_data:
        return {}
    
    generation_times = [d['generation_time'] for d in llm_data if d['generation_time'] > 0]
    
    return {
        'total_generations': len(llm_data),
        'avg_generation_time': statistics.mean(generation_times) if generation_times else 0,
        'min_generation_time': min(generation_times) if generation_times else 0,
        'max_generation_time': max(generation_times) if generation_times else 0,
        'avg_response_length': statistics.mean(d['response_length'] for d in llm_data),
        'models_used': Counter(d['model'] for d in llm_data)
    }

def analyze_sources(logs: List[Dict]) -> Dict[str, int]:
    """Analyze which sources are being used most."""
    sources = []
    
    for log in logs:
        stages = log.get('stages', {})
        if '3_retrieval' in stages:
            metadata = stages['3_retrieval']['data'].get('retrieval_metadata', {})
            sources.extend(metadata.get('sources', []))
    
    return Counter(sources)

def analyze_grade_levels(logs: List[Dict]) -> Dict[str, int]:
    """Analyze distribution of grade levels."""
    grades = []
    
    for log in logs:
        stages = log.get('stages', {})
        if '1_input' in stages:
            params = stages['1_input']['data'].get('parameters', {})
            grade = params.get('grade_level')
            if grade:
                grades.append(grade)
    
    return Counter(grades)

def analyze_errors(logs: List[Dict]) -> List[Dict[str, Any]]:
    """Find and analyze errors."""
    errors = []
    
    for log in logs:
        stages = log.get('stages', {})
        for stage_name, stage_data in stages.items():
            if 'error' in stage_name:
                errors.append({
                    'session_id': log['session_id'],
                    'session_type': log['session_type'],
                    'timestamp': log['timestamp'],
                    'error_type': stage_data['data'].get('error_type'),
                    'error_message': stage_data['data'].get('error_message'),
                    'stage': stage_name
                })
    
    return errors

def analyze_query_patterns(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze query patterns."""
    queries = []
    
    for log in logs:
        stages = log.get('stages', {})
        if '1_input' in stages:
            query = stages['1_input']['data'].get('original_query', '')
            queries.append(query)
    
    if not queries:
        return {}
    
    query_lengths = [len(q) for q in queries]
    
    # Count common words
    words = []
    for query in queries:
        words.extend(query.lower().split())
    
    return {
        'total_queries': len(queries),
        'avg_query_length': statistics.mean(query_lengths),
        'min_query_length': min(query_lengths),
        'max_query_length': max(query_lengths),
        'most_common_words': Counter(words).most_common(10)
    }

def analyze_success_rate(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze success vs failure rate."""
    success_count = 0
    failure_count = 0
    
    for log in logs:
        stages = log.get('stages', {})
        if '6_final_output' in stages:
            success = stages['6_final_output']['data']['response'].get('success', False)
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    total = success_count + failure_count
    
    return {
        'total_sessions': total,
        'successful': success_count,
        'failed': failure_count,
        'success_rate': (success_count / total * 100) if total > 0 else 0
    }

def print_analysis(logs: List[Dict]):
    """Print comprehensive analysis."""
    
    if not logs:
        print("No logs to analyze!")
        return
    
    print("\n" + "="*70)
    print("PIPELINE LOG ANALYSIS")
    print("="*70)
    print(f"Total Sessions: {len(logs)}")
    print(f"Time Range: {min(log['timestamp'] for log in logs)} to {max(log['timestamp'] for log in logs)}")
    print("="*70)
    
    # Session Types
    print("\n📊 SESSION TYPES")
    print("-" * 70)
    session_types = analyze_session_types(logs)
    for session_type, count in session_types.most_common():
        percentage = (count / len(logs)) * 100
        print(f"  {session_type:<25} {count:>5} ({percentage:>5.1f}%)")
    
    # Success Rate
    print("\n✅ SUCCESS RATE")
    print("-" * 70)
    success_data = analyze_success_rate(logs)
    print(f"  Total Sessions:     {success_data['total_sessions']}")
    print(f"  Successful:         {success_data['successful']}")
    print(f"  Failed:             {success_data['failed']}")
    print(f"  Success Rate:       {success_data['success_rate']:.1f}%")
    
    # Retrieval Performance
    print("\n🔍 RETRIEVAL PERFORMANCE")
    print("-" * 70)
    retrieval_data = analyze_retrieval_performance(logs)
    if retrieval_data:
        print(f"  Total Retrievals:    {retrieval_data['total_retrievals']}")
        print(f"  Avg Chunks:          {retrieval_data['avg_chunks']:.1f}")
        print(f"  Avg Context Length:  {retrieval_data['avg_context_length']:.0f} chars")
        print(f"  Avg Score:           {retrieval_data['avg_score']:.3f}")
        print(f"\n  Strategies Used:")
        for strategy, count in retrieval_data['strategies_used'].items():
            print(f"    {strategy:<20} {count:>5}")
    
    # LLM Performance
    print("\n🤖 LLM PERFORMANCE")
    print("-" * 70)
    llm_data = analyze_llm_performance(logs)
    if llm_data:
        print(f"  Total Generations:   {llm_data['total_generations']}")
        print(f"  Avg Time:            {llm_data['avg_generation_time']:.2f}s")
        print(f"  Min Time:            {llm_data['min_generation_time']:.2f}s")
        print(f"  Max Time:            {llm_data['max_generation_time']:.2f}s")
        print(f"  Avg Response Length: {llm_data['avg_response_length']:.0f} chars")
        print(f"\n  Models Used:")
        for model, count in llm_data['models_used'].items():
            print(f"    {model:<20} {count:>5}")
    
    # Grade Levels
    print("\n📚 GRADE LEVEL DISTRIBUTION")
    print("-" * 70)
    grades = analyze_grade_levels(logs)
    if grades:
        for grade, count in sorted(grades.items()):
            percentage = (count / sum(grades.values())) * 100
            print(f"  {grade:<15} {count:>5} ({percentage:>5.1f}%)")
    else:
        print("  No grade level data available")
    
    # Sources
    print("\n📖 TOP SOURCES")
    print("-" * 70)
    sources = analyze_sources(logs)
    if sources:
        for source, count in sources.most_common(10):
            print(f"  {source:<40} {count:>5}")
    else:
        print("  No source data available")
    
    # Query Patterns
    print("\n💬 QUERY PATTERNS")
    print("-" * 70)
    query_data = analyze_query_patterns(logs)
    if query_data:
        print(f"  Total Queries:      {query_data['total_queries']}")
        print(f"  Avg Length:         {query_data['avg_query_length']:.0f} chars")
        print(f"  Min Length:         {query_data['min_query_length']} chars")
        print(f"  Max Length:         {query_data['max_query_length']} chars")
        print(f"\n  Most Common Words:")
        for word, count in query_data['most_common_words']:
            if len(word) > 3:  # Skip short words
                print(f"    {word:<20} {count:>5}")
    
    # Errors
    print("\n❌ ERRORS")
    print("-" * 70)
    errors = analyze_errors(logs)
    if errors:
        print(f"  Total Errors: {len(errors)}\n")
        for error in errors[:5]:  # Show first 5
            print(f"  Session: {error['session_id']}")
            print(f"  Type:    {error['error_type']}")
            print(f"  Message: {error['error_message'][:60]}...")
            print()
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    else:
        print("  No errors found! 🎉")
    
    print("\n" + "="*70)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Pipeline Logs")
    parser.add_argument("--log-dir", "-d", default="logs/pipeline_traces", help="Log directory")
    parser.add_argument("--export", "-e", help="Export analysis to JSON file")
    
    args = parser.parse_args()
    
    # Load logs
    print(f"\nLoading logs from: {args.log_dir}")
    logs = load_logs(args.log_dir)
    
    if not logs:
        print("No logs found!")
        return
    
    print(f"Loaded {len(logs)} log files")
    
    # Analyze and print
    print_analysis(logs)
    
    # Export if requested
    if args.export:
        analysis = {
            'session_types': dict(analyze_session_types(logs)),
            'success_rate': analyze_success_rate(logs),
            'retrieval_performance': analyze_retrieval_performance(logs),
            'llm_performance': analyze_llm_performance(logs),
            'grade_levels': dict(analyze_grade_levels(logs)),
            'sources': dict(analyze_sources(logs)),
            'query_patterns': analyze_query_patterns(logs),
            'errors': analyze_errors(logs)
        }
        
        with open(args.export, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Analysis exported to: {args.export}")

if __name__ == "__main__":
    main()