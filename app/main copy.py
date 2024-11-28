import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from embeddings.embedder_gpu import Embedder
from model.ollama_integration import OllamaLLM
import torch
import time
torch.cuda.empty_cache()
import gc
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 메모리 비우기
clear_memory()


# Ensure the base directory is in sys.path

base_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)  # Add the project root directory to sys.path

# Debugging: Print sys.path and base_dir
print("Base Directory:", base_dir)
print("sys.path:", sys.path)

# Define paths
data_path = os.path.join(base_dir, "data", "articles.json")



def main():
    try:
        # Load query and data
        query = "시가 총액이란?"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item["title"] + " " + item["content"] for item in data]

        # Generate embeddings
        embedder = Embedder(device="cuda:0")  # Use the first GPU
        print("Generating embeddings...")
        embeddings = embedder.generate_embeddings(texts)

        # Create GPU-based FAISS index
        print("Creating GPU-based FAISS index...")
        index = embedder.create_gpu_faiss_index(embeddings)
        torch.cuda.empty_cache()

        # Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedder.generate_embeddings([query])[0]

        # Search for relevant documents
        print("Searching FAISS index...")
        distances, indices = embedder.search(index, query_embedding, top_k=5)
        torch.cuda.empty_cache()
        time.sleep(5)

        # Prepare prompt for LLM
        prompt = "The following are relevant articles:\n"
        for idx in indices[0]:
            prompt += f"- {data[idx]['title']}: {data[idx]['content'][:200]}...\n"
        prompt += "\nWhat is market capitalization?"
        torch.cuda.empty_cache()

        # Query LLM using OllamaLLM class
        print("Querying LLM...")
        llm = OllamaLLM()  # Create an instance of OllamaLLM
        response = llm.query(prompt, streaming=True) 
        print("\nLLM Response:\n", response)
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()















# def main():
#     try:
#         # Load query and data
#         query = "How does Bitcoin affect the stock market?"
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"Data file not found: {data_path}")

#         # Load data
#         with open(data_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         texts = [item["title"] + " " + item["content"] for item in data]

#         # Generate embeddings
#         embedder = Embedder(device="cuda:0")  # Use the first GPU # Use GPU for embedding
#         print("Generating embeddings...")
#         embeddings = embedder.generate_embeddings(texts)

#         # Create GPU-based FAISS index
#         print("Creating GPU-based FAISS index...")
#         index = embedder.create_gpu_faiss_index(embeddings)

#         # Generate query embedding
#         print("Generating query embedding...")
#         query_embedding = embedder.generate_embeddings([query])[0]

#         # Search for relevant documents
#         print("Searching FAISS index...")
#         distances, indices = embedder.search(index, query_embedding, top_k=5)

#         # Prepare prompt for LLM
#         prompt = "The following are relevant articles:\n"
#         for idx in indices[0]:
#             prompt += f"- {data[idx]['title']}: {data[idx]['content'][:200]}...\n"
#         prompt += "\nSummarize the relationship between Bitcoin and the stock market."

#         # Query LLM
#         print("Querying LLM...")
#         llm = OllamaLLM()
#         response = llm.query(prompt)
#         print("\nLLM Response:\n", response)

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()