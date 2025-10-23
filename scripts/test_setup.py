#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.elasticsearch_client import GeometryElasticsearchClient

def test_setup():
    es_client = GeometryElasticsearchClient()
    
    # Check index exists
    if es_client.client.indices.exists(index=es_client.index_name):
        print("✓ Elasticsearch index created")
        
        # Get document count
        count = es_client.client.count(index=es_client.index_name)['count']
        print(f"✓ Total chunks indexed: {count}")
        
        # Sample search
        results = es_client.client.search(
            index=es_client.index_name,
            body={
                "query": {"match": {"text": "triangle"}},
                "size": 3
            }
        )
        print(f"✓ Sample search for 'triangle': {results['hits']['total']['value']} results")
    else:
        print("✗ Index not found")

if __name__ == "__main__":
    test_setup()