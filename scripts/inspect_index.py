import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from elasticsearch import Elasticsearch
from src.config.settings import settings
import json

# Connect to Elasticsearch
es = Elasticsearch(
    hosts=[{"host": settings.ES_HOST, "port": settings.ES_PORT, "scheme": settings.ES_SCHEME}],
    request_timeout=settings.ES_TIMEOUT
)

INDEX_NAME = settings.ES_INDEX_NAME  # "geometry_k12_rag"

# Fetch 3 sample vector entries
response = es.search(
    index=INDEX_NAME,
    body={
        "query": {"exists": {"field": "embedding"}},
        "_source": [
            "chunk_id", "level", "doc_id", "text",
            "embedding", "grade_level", "difficulty",
            "contains_theorem", "contains_formula"
        ],
        "size": 3
    }
)

print("="*80)
print(f"🔍 Sample entries from '{INDEX_NAME}' index:")
print("="*80)

for hit in response["hits"]["hits"]:
    doc = hit["_source"]
    # Truncate long text for readability
    text_preview = doc["text"][:200].replace("\n", " ") + "..."
    print(json.dumps({**doc, "text": text_preview}, indent=2))
    print("-"*80)
