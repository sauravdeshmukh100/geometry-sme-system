from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Configuration settings for the Geometry SME system."""
    
    # Elasticsearch Configuration
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_INDEX_NAME: str = "geometry_k12_rag"
    ES_TIMEOUT: int = 30
    ES_SCHEME: str = "http"  # required for Elasticsearch 8.x

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    embedding_max_tokens: int = 384  # Added new parameter
    device: str = "cuda"
    
    # Chunking Configuration
    CHUNK_SIZE_LEVEL_0: int = 2048
    CHUNK_SIZE_LEVEL_1: int = 350
    CHUNK_SIZE_LEVEL_2: int = 100
    CHUNK_OVERLAP: int = 20
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Paths
    data_dir: Path = Path("../data")  # Updated path
    raw_data_dir: Path = Path("../data/raw")  # Updated path
    processed_data_dir: Path = Path("../data/processed")  # Updated path
    metadata_dir: Path = Path("../data/metadata")  # Updated path

    # Logging
    log_level: str = "INFO"
    log_file: str = "../logs/geometry_sme.log"  # Updated to lowercase and relative path

    # Reranker Configuration
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = True

    # Cache Configuration
    EMBEDDING_CACHE_TTL: int = 3600

    # Retrieval Configuration
    DEFAULT_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    VECTOR_WEIGHT: float = 0.7
    KEYWORD_WEIGHT: float = 0.3

    # Phase 3: Gemini API Configuration
    GEMINI_API_KEY: str = "AIzaSyCHkdJ94erVAdUBB1Dt1M9ODe-5ndKG2VY"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.95
    
    # Phase 3: Email Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    EMAIL_USERNAME: str = "sauravdeshmukh200@gmail.com"
    EMAIL_PASSWORD: str = "oldu qwnv snmm oxxr"
    SMTP_FROM_EMAIL: str = "sauravdeshmukh200@gmail.com"

    # JWT Authentication Configuration (NEW)
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-make-it-very-long-and-random")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    
    # Password Requirements
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    
    # User Management
    DEFAULT_USER_ROLE: str = "student"
    ALLOW_USER_REGISTRATION: bool = True
    REQUIRE_EMAIL_VERIFICATION: bool = False  # Set to True in production

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