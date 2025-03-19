from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Define the index mapping
index_name = "plagiarism_index2"
index_mapping = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 384,  # Match embedding model dimensions
                "index": True,  
                "similarity": "cosine"  # Use cosine similarity for search
            },
            "document_text": {
                "type": "text"  
            }
        }
    }
}

# Create the index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_mapping)
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")
