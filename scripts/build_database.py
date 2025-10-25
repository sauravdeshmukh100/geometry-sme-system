#!/usr/bin/env python
"""
Build database script - Process documents and index to Elasticsearch.
Updated to pass document metadata to chunks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.data_preparation.document_processor import GeometryDocumentProcessor
from src.data_preparation.chunk_manager import GeometryChunkManager
from src.database.elasticsearch_client import GeometryElasticsearchClient

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

def main():
    """Main database building function."""
    print("="*70)
    print("GEOMETRY SME - DATABASE BUILD")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    processor = GeometryDocumentProcessor()
    chunk_manager = GeometryChunkManager()
    es_client = GeometryElasticsearchClient()
    
    # Load embedding model
    print(f"\n2. Loading embedding model: {settings.embedding_model}")
    try:
        embedder = SentenceTransformer(settings.embedding_model, device=settings.device)
        print(f"   ✓ Model loaded on {settings.device}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return 1
    
    # Create Elasticsearch index
    print("\n3. Creating Elasticsearch index...")
    try:
        es_client.create_index(recreate=True)
        print(f"   ✓ Index '{settings.ES_INDEX_NAME}' created")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return 1
    
    # Process documents
    print(f"\n4. Processing documents from {settings.raw_data_dir}")
    print("   " + "-"*66)
    
    try:
        processed_docs = processor.process_directory(settings.raw_data_dir)
        print(f"   ✓ Processed {len(processed_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        return 1
    
    if not processed_docs:
        print("\n   ⚠️  No documents processed!")
        print("   Please add documents to data/raw/ directory")
        return 1
    
    # Create chunks and index them
    print(f"\n5. Creating chunks and generating embeddings...")
    print("   " + "-"*66)
    
    all_chunks_data = []
    
    for doc in tqdm(processed_docs, desc="   Processing docs"):
        try:
            # Create hierarchical chunks with document metadata
            chunks = chunk_manager.create_hierarchical_chunks(
                doc['content'],
                doc['doc_id'],
                doc['file_name'],
                doc_metadata=doc['metadata']  # Pass metadata to chunks!
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {doc['file_name']}")
                continue
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            
            try:
                embeddings = embedder.encode(
                    chunk_texts, 
                    batch_size=32, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {doc['file_name']}: {e}")
                continue
            
            # Prepare data for indexing
            for chunk, embedding in zip(chunks, embeddings):
                chunk_data = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'parent_id': chunk.parent_id,
                    'text': chunk.text,
                    'level': chunk.level,
                    'source': chunk.source,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'embedding': embedding.tolist(),
                    
                    # Chunk-specific metadata
                    'contains_theorem': chunk.metadata.get('contains_theorem', False),
                    'contains_formula': chunk.metadata.get('contains_formula', False),
                    'contains_shape': chunk.metadata.get('contains_shape', False),
                    'contains_angle': chunk.metadata.get('contains_angle', False),
                    'has_numbers': chunk.metadata.get('has_numbers', False),
                    'topic_density': chunk.metadata.get('topic_density', 0.0),
                    
                    # Document-level metadata (passed through)
                    'grade_level': chunk.metadata.get('grade_level', doc['metadata'].get('grade_level', 'Unknown')),
                    'difficulty': chunk.metadata.get('difficulty', doc['metadata'].get('difficulty', 'Unknown')),
                    'source_type': chunk.metadata.get('source_type', doc.get('source_type', 'Unknown')),
                    'topics': doc['metadata'].get('topics', []),
                    
                    # Chapter info if available
                    'chapter_info': chunk.metadata.get('chapter_info', doc['metadata'].get('chapter_info')),
                    
                    # Additional metadata
                    'metadata': chunk.metadata
                }
                all_chunks_data.append(chunk_data)
        
        except Exception as e:
            logger.error(f"Error processing {doc['file_name']}: {e}", exc_info=True)
            continue
    
    if not all_chunks_data:
        print("\n   ⚠️  No chunks created!")
        return 1
    
    # Index chunks to Elasticsearch
    print(f"\n6. Indexing {len(all_chunks_data)} chunks to Elasticsearch...")
    print("   " + "-"*66)
    
    try:
        es_client.bulk_index(all_chunks_data, batch_size=100)
        print(f"   ✓ Successfully indexed {len(all_chunks_data)} chunks")
    except Exception as e:
        logger.error(f"Failed to index chunks: {e}")
        return 1
    
    # Print statistics
    print(f"\n{'='*70}")
    print("DATABASE BUILD COMPLETE")
    print("="*70)
    
    print(f"\nStatistics:")
    print(f"  Documents processed: {len(processed_docs)}")
    print(f"  Total chunks created: {len(all_chunks_data)}")
    
    # Level distribution
    level_counts = {0: 0, 1: 0, 2: 0}
    for chunk in all_chunks_data:
        level_counts[chunk['level']] = level_counts.get(chunk['level'], 0) + 1
    
    print(f"\n  Chunk Distribution:")
    print(f"    Level 0 (2048 tokens): {level_counts[0]:,} chunks")
    print(f"    Level 1 (512 tokens):  {level_counts[1]:,} chunks")
    print(f"    Level 2 (128 tokens):  {level_counts[2]:,} chunks")
    
    # Grade distribution
    grade_dist = {}
    for chunk in all_chunks_data:
        grade = chunk['grade_level']
        grade_dist[grade] = grade_dist.get(grade, 0) + 1
    
    print(f"\n  Grade Distribution:")
    for grade in sorted(grade_dist.keys()):
        count = grade_dist[grade]
        percentage = (count / len(all_chunks_data)) * 100
        print(f"    {grade:25s}: {count:5,} chunks ({percentage:5.1f}%)")
    
    # Difficulty distribution
    diff_dist = {}
    for chunk in all_chunks_data:
        diff = chunk['difficulty']
        diff_dist[diff] = diff_dist.get(diff, 0) + 1
    
    print(f"\n  Difficulty Distribution:")
    for diff in sorted(diff_dist.keys()):
        count = diff_dist[diff]
        percentage = (count / len(all_chunks_data)) * 100
        print(f"    {diff:25s}: {count:5,} chunks ({percentage:5.1f}%)")
    
    # Content type statistics
    theorem_count = sum(1 for c in all_chunks_data if c.get('contains_theorem', False))
    formula_count = sum(1 for c in all_chunks_data if c.get('contains_formula', False))
    shape_count = sum(1 for c in all_chunks_data if c.get('contains_shape', False))
    
    print(f"\n  Content Statistics:")
    print(f"    Chunks with theorems: {theorem_count:,}")
    print(f"    Chunks with formulas: {formula_count:,}")
    print(f"    Chunks with shapes:   {shape_count:,}")
    
    print(f"\n{'='*70}")
    print("\nNext steps:")
    print("  1. Verify: python scripts/verify_data_processing.py")
    print("  2. Test:   python scripts/test_rag_pipeline.py")
    print("  3. Demo:   python scripts/interactive_retrieval.py")
    print(f"\n{'='*70}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())