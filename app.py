import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
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
    query_embedding = model.encode(query_text).tolist()

    query = {
        "size": top_k,
        "knn": {  
            "field": "embedding",  
            "query_vector": query_embedding,  
            "k": top_k,  
            "num_candidates": 10  
        }
    }

    try:
        response = es.search(index="plagiarism_index2", body=query)
        results = []
        for hit in response["hits"]["hits"]:
            similarity_score = np.dot(query_embedding, hit["_source"]["embedding"])
            results.append((hit["_source"]["document_text"], similarity_score))
        return results
    except Exception as e:
        messagebox.showerror("Error", f"Elasticsearch query failed: {str(e)}")
        return []

def check_plagiarism():
    """
    Get user input text and search for similar documents.
    """
    query_text = text_input.get("1.0", tk.END).strip()
    if not query_text:
        messagebox.showwarning("Warning", "Please enter some text to check.")
        return

    results = find_similar_documents(query_text)

    # Clear previous results
    results_list.delete(0, tk.END)

    if results:
        for i, (doc_text, score) in enumerate(results):
            results_list.insert(tk.END, f"{i+1}. Similarity: {score:.4f}")
            results_list.insert(tk.END, f"   {doc_text[:100]}...\n")
    else:
        results_list.insert(tk.END, "No similar documents found.")

def upload_file():
    """
    Load text from a file and populate the text area.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            text_input.delete("1.0", tk.END)
            text_input.insert(tk.END, file.read())

def store_file_in_db():
    """
    Upload a text file, convert it into an embedding, and store in Elasticsearch.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read().strip()  # Trim whitespace and newlines
        
        print(f"File Content:\n{file_text}")  # Debugging print

        if not file_text:  # Check if file is truly empty
            messagebox.showwarning("Warning", "File is empty!")
            return

        file_embedding = model.encode(file_text).tolist()

        # Store in Elasticsearch
        doc = {
            "document_text": file_text,
            "embedding": file_embedding
        }
        es.index(index="plagiarism_index2", document=doc)

        messagebox.showinfo("Success", "File embedded and stored in vector database!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to store file: {str(e)}")


# Initialize Tkinter window
root = tk.Tk()
root.title("AI Plagiarism Detector")
root.geometry("600x500")

# Input text area
tk.Label(root, text="Enter or Upload Text:", font=("Arial", 12)).pack(pady=5)
text_input = tk.Text(root, height=5, width=60)
text_input.pack(pady=5)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="Check Plagiarism", command=check_plagiarism, bg="green", fg="white").grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Upload File", command=upload_file, bg="blue", fg="white").grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Store File in DB", command=store_file_in_db, bg="orange", fg="white").grid(row=0, column=2, padx=5)

# Results display
tk.Label(root, text="Similar Documents:", font=("Arial", 12)).pack(pady=5)
results_list = tk.Listbox(root, width=80, height=10)
results_list.pack(pady=5)

# Run Tkinter event loop
root.mainloop()
