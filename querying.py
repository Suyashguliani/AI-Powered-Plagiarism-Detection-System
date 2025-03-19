import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Load Sentence-BERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

def find_similar_documents(query_text, top_k=3):
    """
    Searches for the most similar documents in Elasticsearch using vector search.
    """
    # Convert query text into an embedding
    query_embedding = model.encode(query_text).tolist()

    # KNN (k-nearest neighbors) search in Elasticsearch
    query = {
    "size": top_k,
    "knn": {  
        "field": "embedding",  # Use "field" instead of "knn"
        "query_vector": query_embedding,  # Use "query_vector"
        "k": top_k,
        "num_candidates": 10
    }
}


    # Perform the search
    response = es.search(index="plagiarism_index2", body=query)

    # Extract results
    results = []
    for hit in response["hits"]["hits"]:
        similarity_score = np.dot(query_embedding, hit["_source"]["embedding"])  # Cosine similarity approximation
        results.append((hit["_source"]["document_text"], similarity_score))

    return results

# Example Query
query_text = "My name is Suyash and I am a software engineer."
similar_docs = find_similar_documents(query_text)

# Print results
print("\n🔍 Similar Documents Found:")
for i, (doc_text, score) in enumerate(similar_docs):
    print(f"{i+1}. Similarity: {score:.4f} → {doc_text[:100]}...")
