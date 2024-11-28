import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import time
import torch
from embeddings.embedder_gpu import Embedder
from model.ollama_integration import OllamaLLM

def truncate_prompt(prompt, max_tokens=2048):
    """프롬프트 길이를 제한"""
    if len(prompt.split()) > max_tokens:
        return " ".join(prompt.split()[:max_tokens])
    return prompt

def retry_query(llm, prompt, retries=3, delay=5):
    """LLM 호출 실패 시 재시도"""
    for attempt in range(retries):
        try:
            response = llm.query(prompt, streaming=True)
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError("All retries failed.")

def main():
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
        data_path = os.path.join(base_dir, "data", "articles.json")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item["title"] + " " + item["content"] for item in data]

        # Generate embeddings
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        embedder = Embedder(device=device)
        print("Generating embeddings...")
        embeddings = embedder.generate_embeddings(texts)

        # Create GPU-based FAISS index
        print("Creating FAISS index...")
        index = embedder.create_gpu_faiss_index(embeddings)
        torch.cuda.empty_cache()

        # Generate query embedding
        query = "What is market capitalization?"
        print("Generating query embedding...")
        query_embedding = embedder.generate_embeddings([query])[0]

        # Search for relevant documents
        print("Searching FAISS index...")
        distances, indices = embedder.search(index, query_embedding, top_k=5)
        torch.cuda.empty_cache()

        # Prepare prompt for LLM
        prompt = "The following are relevant articles:\n"
        for idx in indices[0]:
            prompt += f"- {data[idx]['title']}: {data[idx]['content'][:100]}...\n"
        prompt += "\nExplain market capitalization."
        prompt = truncate_prompt(prompt)

        # Query LLM
        print("Querying LLM...")
        llm = OllamaLLM()
        response = retry_query(llm, prompt)
        print("\nLLM Response:\n", response)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
