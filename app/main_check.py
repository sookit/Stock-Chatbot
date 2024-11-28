import sys
import os
import json

# Ensure the base directory is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Debugging: Print sys.path and base_dir
print("Base Directory:", base_dir)
print("sys.path:", sys.path)

try:
    from embeddings.embedder import Embedder
    from retrieval.retriever import Retriever
    from model.ollama_integration import OllamaLLM
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Current sys.path:")
    for path in sys.path:
        print(path)
    raise

# Define paths
data_path = os.path.join(base_dir, "data", "articles.json")
index_path = os.path.join(base_dir, "data", "faiss_index.bin")

def main():
    try:
        # Load query and data
        query = "How does Bitcoin affect the stock market?"
        
        # Check if required files exist
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        # Generate query embedding
        embedder = Embedder()
        query_embedding = embedder.generate_embeddings([query])[0]

        # Retrieve relevant documents
        retriever = Retriever(index_path, data_path)
        results = retriever.retrieve(query_embedding)

        if not results:
            raise ValueError("No relevant documents retrieved from the index.")

        # Prepare prompt for LLM
        prompt = "The following are relevant articles:\n"
        for result in results:
            prompt += f"- {result['title']}: {result['content'][:200]}...\n"
        prompt += "\nSummarize the relationship between Bitcoin and the stock market."

        # Query LLM
        llm = OllamaLLM()
        response = llm.query(prompt)
        print("LLM Response:\n", response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
