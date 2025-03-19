from sentence_transformers import SentenceTransformer

# Load the Sentence-BERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example text
text = "Artificial Intelligence is revolutionizing plagiarism detection."

# Generate embeddings (vector representation)
embedding = model.encode(text).tolist()

print(f"Embedding size: {len(embedding)}")  # Should be 384 dimensions
print(f"Sample Embedding: {embedding[:5]}...")  # Print first 5 values
