import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, request, jsonify
import torch
import gc
import time
from embeddings.embedder_gpu_all_data import Embedder
from model.ollama_integration import OllamaLLM
from data.naver_news_time_crawler import get_articles_info_with_time_limit, save_articles_to_json

app = Flask(__name__)

# 전역 변수 초기화
llm = OllamaLLM()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedder = Embedder(device=device)
faiss_index = None
articles_info = []

def clear_memory():
    """GPU 및 Python 메모리 정리"""
    torch.cuda.empty_cache()
    gc.collect()

def truncate_prompt(prompt, max_tokens=2048):
    """프롬프트 길이를 제한"""
    if len(prompt.split()) > max_tokens:
        return " ".join(prompt.split()[:max_tokens])
    return prompt

def retry_query(llm, prompt, retries=3, delay=5):
    """LLM 호출 실패 시 재시도"""
    for attempt in range(retries):
        try:
            response = llm.query(prompt, streaming=False)
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                clear_memory()
                time.sleep(delay)
    raise RuntimeError("All retries failed.")

@app.route('/crawl', methods=['POST'])
def crawl_news():
    global articles_info, faiss_index
    data = request.json
    query = data.get("query")
    time_limit = data.get("time_limit", 60)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # 뉴스 크롤링
    try:
        articles_info = get_articles_info_with_time_limit(query, max_time=time_limit)
        if not articles_info:
            return jsonify({"error": "No articles fetched"}), 404

        save_articles_to_json(articles_info, query)

        # 임베딩 생성
        texts = [item["title"] + " " + item["content"] for item in articles_info]
        embeddings = embedder.generate_embeddings(texts)

        # FAISS 인덱스 생성
        faiss_index = embedder.create_gpu_faiss_index(embeddings)
        clear_memory()

        return jsonify({"message": "Crawling and indexing completed", "articles_count": len(articles_info)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_llm():
    global articles_info, faiss_index
    data = request.json
    analysis_query = data.get("query")

    if not analysis_query:
        return jsonify({"error": "Query is required"}), 400

    if not articles_info or not faiss_index:
        return jsonify({"error": "No articles indexed. Please crawl news first."}), 400

    try:
        # FAISS 검색
        query_embedding = embedder.generate_embeddings([analysis_query])[0]
        distances, indices = embedder.search(faiss_index, query_embedding, top_k=5)
        clear_memory()

        # LLM 프롬프트 생성
        prompt = "Explain shortly, the following are relevant articles:\n"
        for idx in indices[0]:
            prompt += f"- {articles_info[idx]['title']}: {articles_info[idx]['content'][:200]}...\n"
        prompt += f"\n{analysis_query}"
        prompt = truncate_prompt(prompt)

        # LLM 호출
        response = retry_query(llm, prompt)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the LLM API"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
