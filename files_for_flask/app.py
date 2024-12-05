import time
import torch
import gc
from embedder_gpu_all_data import Embedder
from ollama_integration import OllamaLLM  # OllamaLLM 클래스 임포트
from naver_news_time_crawler import get_articles_info_with_time_limit, save_articles_to_json
from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS를 임포트

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
            response = llm.query(prompt)  # prompt 전달
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                clear_memory()
                time.sleep(delay)
    raise RuntimeError("All retries failed.")

app = Flask(__name__)
CORS(app)

# 전역 변수 선언
articles_info = []
index = None
embedder = None

@app.route('/api/query', methods=['POST'])
def set_query_and_time_limit():
    global articles_info, embedder, index  # 전역 변수 사용을 명시적으로 선언
    try:
        # 클라이언트로부터 query와 time_limit을 동시에 받음
        user_input = request.json
        query = user_input.get('query')
        time_limit = int(user_input.get('time_limit'))

        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not time_limit:
            return jsonify({"error": "Time limit is required"}), 400
        
        # 뉴스 크롤링
        articles_info = get_articles_info_with_time_limit(query, max_time=time_limit)
        if not articles_info:
            return jsonify({"error": "No articles fetched."}), 400

        # 데이터 저장
        save_articles_to_json(articles_info, query)

        # 데이터 로드 및 임베딩 생성
        print("Loading article data...")
        texts = [item["title"] + " " + item["content"] for item in articles_info]

        # 임베딩 생성
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        embedder = Embedder(device=device)
        print("Generating embeddings...")
        embeddings = embedder.generate_embeddings(texts)

        # FAISS 인덱스 생성
        print("Creating FAISS index...")
        index = embedder.create_gpu_faiss_index(embeddings)
        clear_memory()

        return jsonify({"message": "Crawling and embedding successful. Ready for analysis."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global embedder, index, articles_info  # 전역 변수 사용을 명시적으로 선언
    try:
        user_input = request.json
        analysis_query = user_input.get('analysis_query')
        print(analysis_query)

        if not analysis_query:
            return jsonify({"error": "No analysis query provided."}), 400

        # embedder와 index가 초기화되었는지 확인
        if embedder is None or index is None:
            return jsonify({"error": "Embedding or FAISS index not initialized. Please run the query first."}), 500

        # FAISS 검색
        print("Generating query embedding...")
        query_embedding = embedder.generate_embeddings([analysis_query])[0]
        print("Searching FAISS index...")
        distances, indices = embedder.search(index, query_embedding, top_k=5)
        clear_memory()

        if len(indices) == 0 or len(indices[0]) == 0:
            return jsonify({"error": "No relevant articles found in FAISS search."}), 404

        # LLM 프롬프트 생성
        prompt = "Explain shortly, the following are relevant articles:\n"
        for idx in indices[0]:
            prompt += f"- {articles_info[idx]['title']}: {articles_info[idx]['content'][:200]}...\n"
        prompt += f"\n{analysis_query}"
        prompt = truncate_prompt(prompt)

        # LLM 호출
        print("Querying LLM...")
        try:
            response = retry_query(OllamaLLM(), prompt)  # OllamaLLM 인스턴스를 생성 후 prompt 전달
            print("\nLLM Response:\n", response)
            return jsonify({"response": response}), 200
        except Exception as e:
            print(f"Error during LLM query: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"Error during LLM querybb: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
