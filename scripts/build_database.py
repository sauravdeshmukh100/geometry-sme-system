#!/usr/bin/env python
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

# Logging setup
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    processor = GeometryDocumentProcessor()
    chunk_manager = GeometryChunkManager()
    es_client = GeometryElasticsearchClient()
    
    # Load embedding model
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embedder = SentenceTransformer(settings.embedding_model, device=settings.device)
    
    # Create Elasticsearch index
    logger.info("Creating Elasticsearch index...")
    es_client.create_index(recreate=True)
    
    # Process documents
    logger.info(f"Processing documents from {settings.raw_data_dir}")
    processed_docs = processor.process_directory(settings.raw_data_dir)
    
    # Create chunks and index them
    all_chunks_data = []
    
    for doc in tqdm(processed_docs, desc="Creating chunks"):
        chunks = chunk_manager.create_hierarchical_chunks(
            text=doc["content"],
            doc_id=doc["doc_id"],
            source=doc["file_name"]
        )
        logger.debug(f"Chunks returned for {doc['file_name']}: {len(chunks)}")
        if len(chunks) == 0:
            logger.warning(f"No chunks generated for {doc['file_name']} — check chunk_manager logic.")

        
        # Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(chunk_texts, batch_size=32, show_progress_bar=False)
        
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
                'metadata': chunk.metadata,
                'grade_level': doc['metadata'].get('grade_level'),
                'difficulty': doc['metadata'].get('difficulty'),
                'topics': doc['metadata'].get('topics', [])
            }
            all_chunks_data.append(chunk_data)
    
    # Index chunks to Elasticsearch
    logger.info(f"Indexing {len(all_chunks_data)} chunks to Elasticsearch...")
    es_client.bulk_index(all_chunks_data)
    
    logger.info("Database build complete!")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Documents processed: {len(processed_docs)}")
    print(f"- Total chunks created: {len(all_chunks_data)}")
    print(f"- Level 0 chunks: {sum(1 for c in all_chunks_data if c['level'] == 0)}")
    print(f"- Level 1 chunks: {sum(1 for c in all_chunks_data if c['level'] == 1)}")
    print(f"- Level 2 chunks: {sum(1 for c in all_chunks_data if c['level'] == 2)}")

if __name__ == "__main__":
    main()