import faiss
import numpy as np
import json

class Retriever:
    def __init__(self, index_path, data_path):
        self.index = faiss.read_index(index_path)
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def retrieve(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        results = [self.data[i] for i in indices[0]]
        return results

# Example usage
if __name__ == "__main__":
    retriever = Retriever("../data/faiss_index.bin", "../data/articles.json")
    query_embedding = np.random.rand(384)  # Replace with actual query embedding
    results = retriever.retrieve(query_embedding)
    for result in results:
        print(result["title"])
