import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
import re
from bs4 import BeautifulSoup

# 텍스트 전처리 함수 (HTML 태그 및 불필요한 기호 제거)
def clean_text(text):
    clean = BeautifulSoup(text, "html.parser").get_text()
    clean = re.sub(r'[■▲▶●]', '', clean)  # 특정 기호 제거
    clean = re.sub(r'[\n\r\t]', ' ', clean)  # 줄바꿈, 탭 제거
    clean = re.sub(r'\s+', ' ', clean).strip()  # 다중 공백 제거
    clean = re.sub(r'[^\w\s]', '', clean)  # 특수문자 제거
    return clean

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device  # 'cuda' for GPU, 'cpu' for CPU
        self.model.to(self.device)  # Move model to GPU or CPU

    def generate_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Compute mean pooling and move result to CPU for further processing
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)
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


def load_all_articles(data_folder):
    """Load all articles from articles_n.json files in the given folder."""
    articles = []
    for filename in os.listdir(data_folder):
        if filename.startswith("articles_") and filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    # Combine title and content, apply text cleaning
                    clean_title = clean_text(item["title"])
                    clean_content = clean_text(item["content"])
                    articles.append(clean_title + " " + clean_content)
    return articles


# Example usage
if __name__ == "__main__":
    # Define the folder containing articles
    data_folder = "C:/Users/wendy/1_RAG_project/data/articles_data"
    
    # Load all articles from the folder
    print("Loading articles...")
    texts = load_all_articles(data_folder)
    print(f"Total articles loaded: {len(texts)}")

    # Generate embeddings
    print("Generating embeddings...")
    embedder = Embedder(device="cuda")  # Specify 'cuda' for GPU
    embeddings = embedder.generate_embeddings(texts)

    # Create GPU-based FAISS index
    print("Creating FAISS index...")
    index = embedder.create_gpu_faiss_index(embeddings)

    # Query the FAISS index
    query = "하나은행"
    print("Generating query embedding...")
    clean_query = clean_text(query)  # Clean the query text
    query_embedding = embedder.generate_embeddings([clean_query])[0]

    print("Searching FAISS index...")
    distances, indices = embedder.search(index, query_embedding, top_k=5)

    # Print results
    print("Top 5 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Result {i + 1}: Distance={dist}, Index={idx}, Content={texts[idx][:100]}")
