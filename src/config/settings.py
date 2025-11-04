from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Configuration settings for the Geometry SME system."""
    
    # Elasticsearch
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_INDEX_NAME: str = "geometry_k12_rag"
    ES_TIMEOUT: int = 30
    ES_SCHEME: str = "http"  # required for Elasticsearch 8.x
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    device: str = "cuda"
    
    # Chunking Configuration
    chunk_size_level_0: int = 2048
    chunk_size_level_1: int = 350
    chunk_size_level_2: int = 100
    chunk_overlap: int = 20
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Paths
    data_dir: Path = Path("../data")
    raw_data_dir: Path = Path("../data/raw")
    processed_data_dir: Path = Path("../data/processed")
    metadata_dir: Path = Path("../data/metadata")

    # Logging
    log_level: str = "INFO"
    log_file: str = "../logs/geometry_sme.log"

    reranker_model: str = "BAAI/bge-reranker-base"
    enable_reranker: bool = True
    embedding_cache_ttl: int = 3600
    default_top_k: int = 10
    rerank_top_k: int = 5
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
        
    # Geometry-specific settings
    geometry_topics: list = [
        "shapes", "angles", "theorems", "proofs", "triangles",
        "quadrilaterals", "circles", "polygons", "coordinate_geometry",
        "transformations", "similarity", "congruence", "area", "volume"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # allows extra vars if not declared

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

settings = Settings()