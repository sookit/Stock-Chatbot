import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import json


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    def create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def save_faiss_index(self, index, filepath):
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create directory if it doesn't exist
        faiss.write_index(index, filepath)

    def load_faiss_index(self, filepath):
        return faiss.read_index(filepath)


# Example usage
if __name__ == "__main__":
    # Load data
    with open("./data/articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["title"] + " " + item["content"] for item in data]

    # Generate embeddings and save FAISS index
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(texts)
    index = embedder.create_faiss_index(embeddings)
    embedder.save_faiss_index(index, "./data/faiss_index.bin")
    print("FAISS index saved.")
