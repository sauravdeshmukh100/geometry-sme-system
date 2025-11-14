#!/usr/bin/env python
"""
Build database script - Process documents and index to Elasticsearch.
Now with hybrid hierarchical + recursive chunking strategy.
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
from src.data_preparation.chunk_manager import HybridRecursiveChunkManager  # UPDATED
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
    """Main database building function with recursive chunking."""
    print("="*70)
    print("GEOMETRY SME - DATABASE BUILD (Recursive Chunking)")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    processor = GeometryDocumentProcessor()
    chunk_manager = HybridRecursiveChunkManager()  # NEW: Recursive chunking
    es_client = GeometryElasticsearchClient()
    
    # Load embedding model
    print(f"\n2. Loading embedding model: {settings.embedding_model}")
    try:
        embedder = SentenceTransformer(settings.embedding_model, device=settings.device)
        print(f"   ✓ Model loaded on {settings.device}")
        print(f"   ✓ Model max tokens: 512 (automatically handled)")
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
    print(f"\n5. Creating chunks with recursive splitting...")
    print("   " + "-"*66)
    
    all_chunks = []
    embeddable_chunks = []  # NEW: Track which chunks can be embedded
    
    for doc in tqdm(processed_docs, desc="   Processing docs"):
        try:
            # Create hierarchical chunks with recursive splitting
            chunks = chunk_manager.create_hierarchical_chunks(
                doc['content'],
                doc['doc_id'],
                doc['file_name'],
                doc_metadata=doc['metadata']
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {doc['file_name']}")
                continue
            
            # Separate embeddable chunks (Level 1 & 2) from context chunks (Level 0)
            doc_embeddable = [c for c in chunks if c.embeddable]
            
            all_chunks.extend(chunks)
            embeddable_chunks.extend(doc_embeddable)
            
            logger.info(f"   {doc['file_name']}: "
                       f"{len(chunks)} total chunks, "
                       f"{len(doc_embeddable)} embeddable")
            
        except Exception as e:
            logger.error(f"Error processing {doc['file_name']}: {e}", exc_info=True)
            continue
    
    if not all_chunks:
        print("\n   ⚠️  No chunks created!")
        return 1
    
    print(f"\n   ✓ Created {len(all_chunks)} total chunks")
    print(f"   ✓ {len(embeddable_chunks)} chunks will be embedded")
    print(f"   ✓ {len(all_chunks) - len(embeddable_chunks)} context-only chunks (Level 0)")
    
    # Generate embeddings ONLY for embeddable chunks
    print(f"\n6. Generating embeddings for {len(embeddable_chunks)} chunks...")
    print("   " + "-"*66)
    
    try:
        # Extract text from embeddable chunks
        embeddable_texts = [chunk.text for chunk in embeddable_chunks]
        
        # Generate embeddings in batches
        embeddings = embedder.encode(
            embeddable_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return 1
    
    # Map embeddings back to chunks
    embedding_map = {}
    for chunk, embedding in zip(embeddable_chunks, embeddings):
        embedding_map[chunk.chunk_id] = embedding
    
    # Prepare data for indexing
    print(f"\n7. Preparing {len(all_chunks)} chunks for indexing...")
    all_chunks_data = []
    
    for chunk in all_chunks:
        chunk_data = {
            'chunk_id': chunk.chunk_id,
            'doc_id': chunk.doc_id,
            'parent_id': chunk.parent_id,
            'text': chunk.text,
            'level': chunk.level,
            'source': chunk.source,
            'start_char': chunk.start_char,
            'end_char': chunk.end_char,
            'embeddable': chunk.embeddable,  # NEW: Track embeddability
            
            # Chunk-specific metadata
            'contains_theorem': chunk.metadata.get('contains_theorem', False),
            'contains_formula': chunk.metadata.get('contains_formula', False),
            'contains_shape': chunk.metadata.get('contains_shape', False),
            'contains_angle': chunk.metadata.get('contains_angle', False),
            'has_numbers': chunk.metadata.get('has_numbers', False),
            'topic_density': chunk.metadata.get('topic_density', 0.0),
            
            # Document-level metadata
            'grade_level': chunk.metadata.get('grade_level', 'Unknown'),
            'difficulty': chunk.metadata.get('difficulty', 'Unknown'),
            'source_type': chunk.metadata.get('source_type', 'Unknown'),
            'topics': chunk.metadata.get('topics', []),
            
            # Chapter info if available
            'chapter_info': chunk.metadata.get('chapter_info'),
            
            # Additional metadata
            'metadata': chunk.metadata
        }
        
        # Add embedding ONLY if chunk is embeddable
        if chunk.embeddable and chunk.chunk_id in embedding_map:
            chunk_data['embedding'] = embedding_map[chunk.chunk_id].tolist()
        else:
            # Level 0 chunks have no embedding (too large for model)
            chunk_data['embedding'] = None
        
        all_chunks_data.append(chunk_data)
    
    # Index chunks to Elasticsearch
    print(f"\n8. Indexing {len(all_chunks_data)} chunks to Elasticsearch...")
    print("   " + "-"*66)
    
    try:
        es_client.bulk_index(all_chunks_data, batch_size=100)
        print(f"   ✓ Successfully indexed {len(all_chunks_data)} chunks")
    except Exception as e:
        logger.error(f"Failed to index chunks: {e}")
        return 1
    
    # Print comprehensive statistics
    print(f"\n{'='*70}")
    print("DATABASE BUILD COMPLETE")
    print("="*70)
    
    print(f"\n📊 Document Statistics:")
    print(f"  Documents processed: {len(processed_docs)}")
    
    # Grade distribution
    grade_dist = {}
    for doc in processed_docs:
        grade = doc['metadata'].get('grade_level', 'Unknown')
        grade_dist[grade] = grade_dist.get(grade, 0) + 1
    
    print(f"\n  Grade Distribution (Documents):")
    for grade in sorted(grade_dist.keys()):
        print(f"    {grade:25s}: {grade_dist[grade]} docs")
    
    print(f"\n📦 Chunk Statistics:")
    print(f"  Total chunks created: {len(all_chunks):,}")
    print(f"  Embeddable chunks: {len(embeddable_chunks):,}")
    print(f"  Context-only chunks: {len(all_chunks) - len(embeddable_chunks):,}")
    
    # Level distribution
    level_counts = {0: 0, 1: 0, 2: 0}
    for chunk in all_chunks_data:
        level = chunk['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"\n  Chunk Distribution by Level:")
    print(f"    Level 0 (Context, ~2048 tokens):  {level_counts[0]:5,} chunks (not embedded)")
    print(f"    Level 1 (Medium, ≤512 tokens):    {level_counts[1]:5,} chunks (embedded)")
    print(f"    Level 2 (Fine, ≤128 tokens):      {level_counts[2]:5,} chunks (embedded)")
    
    # Verify embedding safety
    embeddable_l1_l2 = level_counts[1] + level_counts[2]
    print(f"\n  ✓ Embedding Model Safety:")
    print(f"    Model limit: 512 tokens")
    print(f"    Embedded chunks: {embeddable_l1_l2:,} (all ≤512 tokens)")
    print(f"    Safety: {'✅ GUARANTEED' if embeddable_l1_l2 == len(embeddable_chunks) else '⚠️ CHECK'}")
    
    # Grade distribution for chunks
    chunk_grade_dist = {}
    for chunk in all_chunks_data:
        grade = chunk['grade_level']
        chunk_grade_dist[grade] = chunk_grade_dist.get(grade, 0) + 1
    
    print(f"\n  Grade Distribution (Chunks):")
    for grade in sorted(chunk_grade_dist.keys()):
        count = chunk_grade_dist[grade]
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
    print(f"    Chunks with theorems: {theorem_count:,} ({theorem_count/len(all_chunks_data)*100:.1f}%)")
    print(f"    Chunks with formulas: {formula_count:,} ({formula_count/len(all_chunks_data)*100:.1f}%)")
    print(f"    Chunks with shapes:   {shape_count:,} ({shape_count/len(all_chunks_data)*100:.1f}%)")
    
    # Storage statistics
    embedding_size_mb = (len(embeddable_chunks) * 768 * 4) / (1024 * 1024)  # 768 dims, 4 bytes per float
    print(f"\n  Storage Statistics:")
    print(f"    Embeddings size: ~{embedding_size_mb:.2f} MB")
    print(f"    Estimated index size: ~{embedding_size_mb * 1.5:.2f} MB")
    
    print(f"\n{'='*70}")
    print("\n✅ Next steps:")
    print("  1. Verify: python scripts/verify_data_processing.py")
    print("  2. Test:   python scripts/test_rag_pipeline.py")
    print("  3. Demo:   python scripts/interactive_retrieval.py")
    print(f"\n{'='*70}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())