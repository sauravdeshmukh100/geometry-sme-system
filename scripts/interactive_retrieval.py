#!/usr/bin/env python
"""
Enhanced Interactive Retrieval Testing Tool
Compatible with improved hierarchical retrieval and adaptive strategies
"""

import sys
import os

# from src.config import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import settings
import logging
from typing import Optional

from src.retrieval.rag_pipeline import (
    GeometryRAGPipeline, 
    RetrievalConfig, 
    RetrievalStrategy
)

# Configure logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InteractiveRetrieval:
    """Enhanced interactive retrieval testing interface."""
    
    def __init__(self):
        print("Initializing Enhanced Geometry RAG Pipeline...")
        try:
            self.pipeline = GeometryRAGPipeline(enable_reranker=True)
            print("✓ Pipeline initialized successfully")
            print("✓ Enhanced features: Adaptive strategy, Level-aware fusion, Deduplication\n")
        except Exception as e:
            print(f"✗ Failed to initialize pipeline: {e}")
            print("  Trying without reranker...")
            try:
                self.pipeline = GeometryRAGPipeline(enable_reranker=False)
                print("✓ Pipeline initialized (without reranker)\n")
            except Exception as e2:
                print(f"✗ Fatal error: {e2}")
                sys.exit(1)
        
        # Default configuration with new features
        self.config = RetrievalConfig(
            strategy=RetrievalStrategy.ADAPTIVE,  # NEW: Use adaptive by default
            top_k=10,
            rerank=True,
            rerank_top_k=5,
            use_metadata_boost=True,
            include_parents=False,
            include_children=False,
            level_preference=None,  # Let adaptive choose
            embeddable_only=True,
            max_context_tokens=4000,  # NEW
            enable_deduplication=True,  # NEW
            similarity_threshold=0.95  # NEW
        )
    
    def print_banner(self):
        """Print welcome banner."""
        print("=" * 70)
        print("  GEOMETRY SME - ENHANCED INTERACTIVE RETRIEVAL")
        print("  Features: Adaptive Strategy | Level-Aware Fusion | Deduplication")
        print("=" * 70)
        print("\nCommands:")
        print("  - Enter a query to search")
        print("  - 'config' to change retrieval settings")
        print("  - 'stats' to show system statistics")
        print("  - 'compare' to compare strategies on a query")
        print("  - 'help' to show this message")
        print("  - 'quit' or 'exit' to quit")
        print("\nCurrent Configuration:")
        self.print_config()
        print()
    
    def print_config(self):
        """Print current configuration with new features."""
        print(f"  Strategy: {self.config.strategy.value}")
        print(f"  Top K: {self.config.top_k}")
        print(f"  Rerank: {self.config.rerank}", end="")
        if self.config.rerank:
            print(f" (Top {self.config.rerank_top_k})")
        else:
            print()
        
        # NEW: Show enhanced features
        print(f"  Level Preference: {self.config.level_preference or 'Auto'}")
        print(f"  Deduplication: {self.config.enable_deduplication}")
        print(f"  Max Context Tokens: {self.config.max_context_tokens}")
        print(f"  Include Parents: {self.config.include_parents}")
        print(f"  Metadata Boost: {self.config.use_metadata_boost}")
        
        if self.config.filters:
            print(f"  Filters: {self.config.filters}")
    
    def configure(self):
        """Enhanced interactive configuration."""
        print("\n" + "─" * 70)
        print("CONFIGURATION")
        print("─" * 70)
        
        # Strategy
        print("\nRetrieval Strategy:")
        print("  1. Vector (semantic search)")
        print("  2. Keyword (BM25)")
        print("  3. Hybrid (vector + keyword)")
        print("  4. Hierarchical (multi-level fusion)")
        print("  5. Adaptive (auto-select based on query) ⭐ NEW")
        
        choice = input("Select strategy (1-5, Enter for current): ").strip()
        if choice == '1':
            self.config.strategy = RetrievalStrategy.VECTOR_ONLY
        elif choice == '2':
            self.config.strategy = RetrievalStrategy.KEYWORD_ONLY
        elif choice == '3':
            self.config.strategy = RetrievalStrategy.HYBRID
        elif choice == '4':
            self.config.strategy = RetrievalStrategy.HIERARCHICAL
        elif choice == '5':
            self.config.strategy = RetrievalStrategy.ADAPTIVE
        
        # Level preference (NEW)
        if self.config.strategy not in [RetrievalStrategy.HIERARCHICAL, RetrievalStrategy.ADAPTIVE]:
            print("\nLevel Preference:")
            print("  0. Level 0 (context, non-searchable)")
            print("  1. Level 1 (medium chunks, ≤384 tokens)")
            print("  2. Level 2 (fine chunks, ≤128 tokens)")
            print("  Auto. Let system decide")
            
            level_choice = input("Select level (0-2, Auto, Enter for Auto): ").strip().lower()
            if level_choice == '0':
                self.config.level_preference = 0
            elif level_choice == '1':
                self.config.level_preference = 1
            elif level_choice == '2':
                self.config.level_preference = 2
            else:
                self.config.level_preference = None
        
        # Top K
        top_k = input(f"\nTop K results ({self.config.top_k}): ").strip()
        if top_k.isdigit():
            self.config.top_k = int(top_k)
        
        # Reranking
        rerank = input(f"Enable reranking? (y/n, current: {'y' if self.config.rerank else 'n'}): ").strip().lower()
        if rerank in ['y', 'yes']:
            self.config.rerank = True
            rerank_k = input(f"  Rerank top K ({self.config.rerank_top_k}): ").strip()
            if rerank_k.isdigit():
                self.config.rerank_top_k = int(rerank_k)
        elif rerank in ['n', 'no']:
            self.config.rerank = False
        
        # NEW: Deduplication
        dedup = input(f"\nEnable deduplication? (y/n, current: {'y' if self.config.enable_deduplication else 'n'}): ").strip().lower()
        if dedup in ['y', 'yes']:
            self.config.enable_deduplication = True
            threshold = input(f"  Similarity threshold ({self.config.similarity_threshold}): ").strip()
            if threshold:
                try:
                    self.config.similarity_threshold = float(threshold)
                except ValueError:
                    pass
        elif dedup in ['n', 'no']:
            self.config.enable_deduplication = False
        
        # NEW: Context tokens
        tokens = input(f"\nMax context tokens ({self.config.max_context_tokens}): ").strip()
        if tokens.isdigit():
            self.config.max_context_tokens = int(tokens)
        
        # NEW: Include parents/children
        parents = input(f"\nInclude parent chunks? (y/n, current: {'y' if self.config.include_parents else 'n'}): ").strip().lower()
        if parents in ['y', 'yes']:
            self.config.include_parents = True
        elif parents in ['n', 'no']:
            self.config.include_parents = False
        
        # Filters
        print("\nFilters (optional):")
        grade = input("  Grade level (Grade 6-10, Enter to skip): ").strip()
        difficulty = input("  Difficulty (Beginner/Intermediate/Advanced, Enter to skip): ").strip()
        
        filters = {}
        if grade:
            if any(str(g) in grade for g in range(6, 11)):
                filters['grade_level'] = grade if 'Grade' in grade else f"Grade {grade}"
        
        if difficulty:
            filters['difficulty'] = difficulty.capitalize()
        
        self.config.filters = filters if filters else None
        
        print("\n✓ Configuration updated:")
        self.print_config()
    
    def show_stats(self):
        """Show enhanced system statistics."""
        print("\n" + "─" * 70)
        print("SYSTEM STATISTICS")
        print("─" * 70)
        
        try:
            stats = self.pipeline.get_statistics()
            
            print(f"\n📊 Index Statistics:")
            print(f"  Total Chunks: {stats.get('total_chunks', 'N/A'):,}")
            print(f"  Embeddable Chunks: {stats.get('embeddable_chunks', 'N/A'):,}")
            print(f"  Context-Only Chunks: {stats.get('context_chunks', 'N/A'):,}")
            print(f"  Index Size: {stats.get('index_size_mb', 0):.2f} MB")
            
            # NEW: Level distribution
            if 'level_distribution' in stats:
                print(f"\n📑 Level Distribution:")
                for level, count in stats['level_distribution'].items():
                    print(f"  {level}: {count:,}")
            
            print(f"\n🔧 Configuration:")
            print(f"  Embedding Model: {stats.get('embedding_model', 'N/A')}")
            print(f"  Embedding Limit: {stats.get('embedding_limit', 'N/A')} tokens")
            print(f"  Reranker: {'✓ Enabled' if stats.get('reranker_enabled') else '✗ Disabled'}")
            print(f"  Cache: {'✓ Enabled' if stats.get('cache_enabled') else '✗ Disabled'}")
            
            # NEW: Enhanced features
            if 'features' in stats:
                print(f"\n✨ Enhanced Features:")
                for feature, enabled in stats['features'].items():
                    status = "✓" if enabled else "✗"
                    feature_name = feature.replace('_', ' ').title()
                    print(f"  {status} {feature_name}")
        
        except Exception as e:
            print(f"\n  ✗ Error getting statistics: {e}")
            logging.error("Stats error", exc_info=True)
    
    def process_query(self, query: str):
        """Process a search query with enhanced output."""
        print("\n" + "─" * 70)
        print(f"Query: {query}")
        print("─" * 70)
        
        try:
            result = self.pipeline.retrieve(query, self.config)
            
            # Summary with new metadata
            print(f"\n✓ Found {len(result.chunks)} chunks")
            print(f"  Strategy Used: {result.metadata.get('strategy_used', 'N/A')}")
            print(f"  Level Selected: {result.metadata.get('level_preference') or 'Auto'}")
            print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
            print(f"  Reranked: {'Yes' if result.metadata.get('reranked') else 'No'}")
            
            # NEW: Enhanced metadata
            print(f"\n📊 Chunk Breakdown:")
            print(f"  Embeddable: {result.metadata.get('embeddable_count', 0)}")
            print(f"  Context-Only: {result.metadata.get('context_only_count', 0)}")
            
            # NEW: Level distribution in results
            level_dist = result.metadata.get('level_distribution', {})
            if level_dist:
                print(f"  Level Distribution: {dict(level_dist)}")
            
            if result.metadata.get('sources'):
                sources = ', '.join(result.metadata['sources'][:3])
                more = len(result.metadata['sources']) - 3
                if more > 0:
                    sources += f" (+{more} more)"
                print(f"\n📚 Sources: {sources}")
            
            if result.metadata.get('topics'):
                topics = ', '.join(result.metadata['topics'][:5])
                print(f"🏷️  Topics: {topics}")
            
            if result.metadata.get('grade_levels'):
                grades = ', '.join(result.metadata['grade_levels'])
                print(f"🎓 Grade Levels: {grades}")
            
            # Show top results with enhanced info
            print(f"\n{'─' * 70}")
            print("TOP RESULTS")
            print("─" * 70)
            
            for i, chunk in enumerate(result.chunks[:5], 1):
                content = chunk.get('content', {})
                
                # Enhanced scoring info
                score = chunk.get('score', 0)
                rerank_score = chunk.get('rerank_score')
                
                print(f"\n[{i}] Score: {score:.4f}", end="")
                if rerank_score is not None:
                    print(f" (Rerank: {rerank_score:.4f})", end="")
                print()
                
                # Level and metadata
                level = content.get('level', 'N/A')
                embeddable = content.get('embeddable', True)
                chunk_type = f"Level {level}" if embeddable else "Context"
                
                print(f"    Type: {chunk_type} | "
                      f"Grade: {content.get('grade_level', 'N/A')} | "
                      f"Difficulty: {content.get('difficulty', 'N/A')}")
                
                # NEW: Special markers
                markers = []
                if content.get('contains_theorem'):
                    markers.append("📘 Theorem")
                if content.get('contains_formula'):
                    markers.append("📐 Formula")
                if content.get('contains_shape'):
                    markers.append("🔷 Shape")
                
                if markers:
                    print(f"    Markers: {', '.join(markers)}")
                
                # Text preview
                text = content.get('text', '')
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"    {preview}")
            
            # Show assembled context with token count
            print(f"\n{'─' * 70}")
            print(f"ASSEMBLED CONTEXT (~{len(result.context.split()) * 1.3:.0f} tokens)")
            print("─" * 70)
            
            context_preview = result.context[:600] + "..." if len(result.context) > 600 else result.context
            print(f"\n{context_preview}")
            
            # Ask if user wants full context
            if len(result.context) > 600:
                show_full = input("\nShow full context? (y/n): ").strip().lower()
                if show_full in ['y', 'yes']:
                    print(f"\n{'─' * 70}")
                    print("FULL CONTEXT")
                    print("─" * 70)
                    print(f"\n{result.context}")
        
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            logging.error("Query processing failed", exc_info=True)
    
    def compare_strategies(self, query: str):
        """NEW: Compare different strategies on the same query."""
        print("\n" + "=" * 70)
        print(f"STRATEGY COMPARISON: {query}")
        print("=" * 70)
        
        strategies = [
            (RetrievalStrategy.ADAPTIVE, "Adaptive"),
            (RetrievalStrategy.HYBRID, "Hybrid"),
            (RetrievalStrategy.HIERARCHICAL, "Hierarchical"),
            (RetrievalStrategy.VECTOR_ONLY, "Vector Only"),
        ]
        
        results = []
        
        for strategy, name in strategies:
            print(f"\n{'─' * 70}")
            print(f"Testing: {name}")
            print("─" * 70)
            
            config = RetrievalConfig(
                strategy=strategy,
                top_k=self.config.top_k,
                rerank=self.config.rerank,
                rerank_top_k=self.config.rerank_top_k
            )
            
            try:
                import time
                start = time.time()
                result = self.pipeline.retrieve(query, config)
                elapsed = time.time() - start
                
                print(f"  Chunks: {len(result.chunks)}")
                print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
                print(f"  Time: {elapsed:.3f}s")
                print(f"  Levels: {result.metadata.get('level_distribution', {})}")
                
                results.append({
                    'strategy': name,
                    'chunks': len(result.chunks),
                    'score': result.metadata.get('avg_score', 0),
                    'time': elapsed
                })
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Summary
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Strategy':<20s} {'Chunks':<10s} {'Avg Score':<12s} {'Time':<10s}")
        print("─" * 70)
        
        for r in sorted(results, key=lambda x: x['score'], reverse=True):
            print(f"{r['strategy']:<20s} {r['chunks']:<10d} {r['score']:<12.4f} {r['time']:<10.3f}s")
    
    def run(self):
        """Run interactive loop."""
        self.print_banner()
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                command = user_input.lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 📐")
                    break
                
                elif command == 'help':
                    self.print_banner()
                
                elif command == 'config':
                    self.configure()
                
                elif command == 'stats':
                    self.show_stats()
                
                elif command == 'compare':
                    query = input("Enter query to compare strategies: ").strip()
                    if query:
                        self.compare_strategies(query)
                    else:
                        print("Query cannot be empty")
                
                else:
                    # Treat as query
                    self.process_query(user_input)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 📐")
                break
            
            except Exception as e:
                print(f"\n✗ Error: {e}")
                logging.error("Unexpected error", exc_info=True)

def main():
    """Main entry point."""
    interactive = InteractiveRetrieval()
    interactive.run()

if __name__ == "__main__":
    main()