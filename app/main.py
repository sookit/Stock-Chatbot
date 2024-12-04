import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import torch
import gc
from embeddings.embedder_gpu_all_data import Embedder
from model.ollama_integration import OllamaLLM
from data.naver_news_time_crawler import get_articles_info_with_time_limit, save_articles_to_json


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
            response = llm.query(prompt, streaming=True)
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                clear_memory()
                time.sleep(delay)
    raise RuntimeError("All retries failed.")


def main():
    try:
        # LLM 초기화
        llm = OllamaLLM()

        # 뉴스 크롤링
        query = input("Enter your query for Naver News crawling: ")
        time_limit = int(input("Enter the time limit for crawling (in seconds): "))
        print(f"Crawling articles for query: {query} with a time limit of {time_limit} seconds...")
        articles_info = get_articles_info_with_time_limit(query, max_time=time_limit)

        if not articles_info:
            print("No articles fetched. Exiting.")
            return

        # 크롤링된 데이터를 저장
        print("Saving crawled articles...")
        save_articles_to_json(articles_info, query)

        # 데이터 로드
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

        # 대화 모드 시작
        print("\n--- Entering Chat Mode ---")
        while True:
            analysis_query = input("\nEnter your analysis query (or type 'exit' to quit): ")
            if analysis_query.lower() == 'exit':
                print("Exiting chat. Goodbye!")
                break

            # FAISS 검색
            print("Generating query embedding...")
            query_embedding = embedder.generate_embeddings([analysis_query])[0]
            print("Searching FAISS index...")
            distances, indices = embedder.search(index, query_embedding, top_k=5)
            clear_memory()

            # LLM 프롬프트 생성
            prompt = "Explain shortly, the following are relevant articles:\n"
            for idx in indices[0]:
                prompt += f"- {articles_info[idx]['title']}: {articles_info[idx]['content'][:200]}...\n"
            prompt += f"\n{analysis_query}"
            prompt = truncate_prompt(prompt)

            # LLM 호출
            print("Querying LLM...")
            try:
                response = retry_query(llm, prompt)
                print("\nLLM Response:\n", response)
            except Exception as e:
                print(f"Error during LLM query: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
