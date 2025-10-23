#!/usr/bin/env python
"""
Interactive retrieval testing tool for Geometry RAG system.
Allows real-time query testing with configurable parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Optional

from src.retrieval.rag_pipeline import (
    GeometryRAGPipeline, 
    RetrievalConfig, 
    RetrievalStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'
)

class InteractiveRetrieval:
    """Interactive retrieval testing interface."""
    
    def __init__(self):
        print("Initializing Geometry RAG Pipeline...")
        try:
            self.pipeline = GeometryRAGPipeline(enable_reranker=True)
            print("✓ Pipeline initialized successfully\n")
        except Exception as e:
            print(f"✗ Failed to initialize pipeline: {e}")
            print("  Trying without reranker...")
            try:
                self.pipeline = GeometryRAGPipeline(enable_reranker=False)
                print("✓ Pipeline initialized (without reranker)\n")
            except Exception as e2:
                print(f"✗ Fatal error: {e2}")
                sys.exit(1)
        
        # Default configuration
        self.config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=10,
            rerank=True,
            rerank_top_k=5,
            use_metadata_boost=True
        )
    
    def print_banner(self):
        """Print welcome banner."""
        print("=" * 70)
        print("  GEOMETRY SME - INTERACTIVE RETRIEVAL TESTING")
        print("=" * 70)
        print("\nCommands:")
        print("  - Enter a query to search")
        print("  - 'config' to change retrieval settings")
        print("  - 'stats' to show system statistics")
        print("  - 'help' to show this message")
        print("  - 'quit' or 'exit' to quit")
        print("\nCurrent Configuration:")
        self.print_config()
        print()
    
    def print_config(self):
        """Print current configuration."""
        print(f"  Strategy: {self.config.strategy.value}")
        print(f"  Top K: {self.config.top_k}")
        print(f"  Rerank: {self.config.rerank}")
        if self.config.rerank:
            print(f"  Rerank Top K: {self.config.rerank_top_k}")
        print(f"  Metadata Boost: {self.config.use_metadata_boost}")
        if self.config.filters:
            print(f"  Filters: {self.config.filters}")
    
    def configure(self):
        """Interactive configuration."""
        print("\n" + "─" * 70)
        print("CONFIGURATION")
        print("─" * 70)
        
        # Strategy
        print("\nRetrieval Strategy:")
        print("  1. Vector (semantic search)")
        print("  2. Keyword (BM25)")
        print("  3. Hybrid (recommended)")
        print("  4. Hierarchical (multi-level)")
        
        choice = input("Select strategy (1-4, Enter for current): ").strip()
        if choice == '1':
            self.config.strategy = RetrievalStrategy.VECTOR_ONLY
        elif choice == '2':
            self.config.strategy = RetrievalStrategy.KEYWORD_ONLY
        elif choice == '3':
            self.config.strategy = RetrievalStrategy.HYBRID
        elif choice == '4':
            self.config.strategy = RetrievalStrategy.HIERARCHICAL
        
        # Top K
        top_k = input(f"Top K results ({self.config.top_k}): ").strip()
        if top_k.isdigit():
            self.config.top_k = int(top_k)
        
        # Reranking
        rerank = input(f"Enable reranking? (y/n, current: {'y' if self.config.rerank else 'n'}): ").strip().lower()
        if rerank in ['y', 'yes']:
            self.config.rerank = True
            rerank_k = input(f"Rerank top K ({self.config.rerank_top_k}): ").strip()
            if rerank_k.isdigit():
                self.config.rerank_top_k = int(rerank_k)
        elif rerank in ['n', 'no']:
            self.config.rerank = False
        
        # Filters
        print("\nFilters (optional):")
        grade = input("  Grade level (Elementary/Middle School/High School, Enter to skip): ").strip()
        difficulty = input("  Difficulty (Beginner/Intermediate/Advanced, Enter to skip): ").strip()
        
        filters = {}
        if grade:
            if 'elementary' in grade.lower():
                filters['grade_level'] = 'Elementary (K-5)'
            elif 'middle' in grade.lower():
                filters['grade_level'] = 'Middle School (6-8)'
            elif 'high' in grade.lower():
                filters['grade_level'] = 'High School (9-12)'
        
        if difficulty:
            filters['difficulty'] = difficulty.capitalize()
        
        self.config.filters = filters if filters else None
        
        print("\n✓ Configuration updated:")
        self.print_config()
    
    def show_stats(self):
        """Show system statistics."""
        print("\n" + "─" * 70)
        print("SYSTEM STATISTICS")
        print("─" * 70)
        
        try:
            stats = self.pipeline.get_statistics()
            print(f"\n  Total Chunks: {stats.get('total_chunks', 'N/A'):,}")
            print(f"  Index Size: {stats.get('index_size_mb', 0):.2f} MB")
            print(f"  Embedding Model: {stats.get('embedding_model', 'N/A')}")
            print(f"  Reranker: {'Enabled' if stats.get('reranker_enabled') else 'Disabled'}")
            print(f"  Cache: {'Enabled' if stats.get('cache_enabled') else 'Disabled'}")
        except Exception as e:
            print(f"\n  ✗ Error getting statistics: {e}")
    
    def process_query(self, query: str):
        """Process a search query."""
        print("\n" + "─" * 70)
        print(f"Query: {query}")
        print("─" * 70)
        
        try:
            result = self.pipeline.retrieve(query, self.config)
            
            # Summary
            print(f"\n✓ Found {len(result.chunks)} chunks")
            print(f"  Strategy: {result.metadata.get('strategy_used', 'N/A')}")
            print(f"  Avg Score: {result.metadata.get('avg_score', 0):.4f}")
            
            if result.metadata.get('sources'):
                sources = ', '.join(result.metadata['sources'][:3])
                print(f"  Sources: {sources}")
            
            if result.metadata.get('topics'):
                topics = ', '.join(result.metadata['topics'][:5])
                print(f"  Topics: {topics}")
            
            if result.metadata.get('grade_levels'):
                grades = ', '.join(result.metadata['grade_levels'])
                print(f"  Grade Levels: {grades}")
            
            # Show top results
            print(f"\n{'─' * 70}")
            print("TOP RESULTS")
            print("─" * 70)
            
            for i, chunk in enumerate(result.chunks[:5], 1):
                content = chunk.get('content', {})
                
                print(f"\n[{i}] Score: {chunk.get('score', 0):.4f}")
                print(f"    Level: {content.get('level', 'N/A')} | "
                      f"Grade: {content.get('grade_level', 'N/A')} | "
                      f"Difficulty: {content.get('difficulty', 'N/A')}")
                
                text = content.get('text', '')
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"    {preview}")
            
            # Show assembled context
            print(f"\n{'─' * 70}")
            print("ASSEMBLED CONTEXT (Preview)")
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
                    print("\nGoodbye!")
                    break
                
                elif command == 'help':
                    self.print_banner()
                
                elif command == 'config':
                    self.configure()
                
                elif command == 'stats':
                    self.show_stats()
                
                else:
                    # Treat as query
                    self.process_query(user_input)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
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