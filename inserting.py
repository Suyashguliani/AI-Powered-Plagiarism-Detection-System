from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define document data
document_text = "Artificial Intelligence is revolutionizing plagiarism detection."
embedding = model.encode(document_text).tolist()

# Index document in Elasticsearch
doc = {
    "document_text": document_text,
    "embedding": embedding
}

response = es.index(index="plagiarism_index2", body=doc)
print(f"Document indexed: {response['_id']}")
