import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import json


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModel.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)  # FP16 사용
        self.device = device  # 'cuda' for GPU, 'cpu' for CPU
        self.model.to(self.device)  # Move model to GPU or CPU

    def generate_embeddings(self, texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.array(embeddings, dtype=np.float32)



    def create_gpu_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        index = faiss.GpuIndexFlatL2(res, dimension)  # GPU-based FAISS index
        index.add(embeddings)  # Add embeddings to the index
        return index

    def search(self, index, query_embedding, top_k=5):
        distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
        return distances, indices


# Example usage
if __name__ == "__main__":
    # Load data
    with open("C:/Users/wendy/1_RAG_project/data/articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["title"] + " " + item["content"] for item in data]

    # Generate embeddings
    embedder = Embedder(device="cuda")  # Specify 'cuda' for GPU
    embeddings = embedder.generate_embeddings(texts)

    # Create GPU-based FAISS index
    index = embedder.create_gpu_faiss_index(embeddings)

    # Query the FAISS index
    query = "How does Bitcoin affect the stock market?"
    query_embedding = embedder.generate_embeddings([query])[0]
    distances, indices = embedder.search(index, query_embedding, top_k=5)

    # Print results
    print("Top 5 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Result {i + 1}: Distance={dist}, Index={idx}, Content={texts[idx][:100]}")
