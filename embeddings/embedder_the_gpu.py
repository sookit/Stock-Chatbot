import faiss
import numpy as np
import ollama

class Embedder:
    def __init__(self, model_name="llama3.1", embedding_dimension=768, device_id=0):
        """
        Embedder 클래스 초기화
        :param model_name: 사용할 모델 이름 (예: "llama3.1")
        :param embedding_dimension: 임베딩 차원 (기본값은 768)
        :param device_id: GPU ID (기본값 0)
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.device_id = device_id
        
        # FAISS GPU 리소스 초기화
        self.res = faiss.StandardGpuResources()
        
        # FAISS 인덱스 초기화 (내적 기반)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, self.device_id, self.index)
        
        # 문서 저장소 초기화
        self.doc_store = []

    def generate_embedding(self, text):
        """
        텍스트를 임베딩하여 벡터 반환
        :param text: 임베딩할 텍스트
        :return: 정규화된 임베딩 벡터
        """
        embedding = ollama.embed(input=text, model=self.model_name)

        # 임베딩 구조가 딕셔너리인 경우 벡터 추출
        if isinstance(embedding, dict):
            embedding_vector = embedding.get('embeddings')
        else:
            embedding_vector = embedding  # 리스트나 배열로 바로 반환되는 경우 처리

        # 벡터가 없으면 None 반환
        if embedding_vector is None:
            return None

        # 리스트 형태인 경우, numpy 배열로 변환
        if isinstance(embedding_vector, list):
            embedding_vector = np.array(embedding_vector)

        # 차원 맞추기 (자르기 또는 패딩)
        if embedding_vector.shape[0] > self.embedding_dimension:
            embedding_vector = embedding_vector[:self.embedding_dimension]
        elif embedding_vector.shape[0] < self.embedding_dimension:
            padding = np.zeros((self.embedding_dimension - embedding_vector.shape[0],))
            embedding_vector = np.concatenate([embedding_vector, padding])

        # L2 정규화
        norm = np.linalg.norm(embedding_vector)
        normalized_embedding = embedding_vector / norm

        return normalized_embedding

    def add_to_faiss(self, item):
        """
        문서를 임베딩하여 FAISS 인덱스에 추가
        :param item: 문서 항목 (title, content 포함)
        """
        combined_text = item['title'] + " " + item['content']
        
        embedding_vector = self.generate_embedding(combined_text)
        if embedding_vector is None:
            print(f"Skipping document: {item['title']}")
            return
        
        # 임베딩 벡터를 2D 배열로 변환 (FAISS에 추가하기 위해)
        embedding_vector = np.array(embedding_vector).reshape(1, -1)
        
        # FAISS 인덱스에 벡터 추가
        self.gpu_index.add(embedding_vector)
        
        # 문서 저장
        self.doc_store.append(item)

    def query_faiss(self, query, top_k=1):
        """
        주어진 질문에 대해 FAISS 인덱스를 사용해 가장 유사한 문서를 검색
        :param query: 검색할 질문
        :param top_k: 상위 K개 문서 반환
        :return: 유사한 문서 목록
        """
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return []

        # 쿼리 임베딩을 2D 배열로 변환 (FAISS 검색을 위해)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # FAISS 인덱스에서 상위 K개 문서 검색
        D, I = self.gpu_index.search(query_embedding, k=top_k)
        
        # 검색된 인덱스를 사용해 원본 문서 반환
        retrieved_docs = [self.doc_store[idx] for idx in I[0]]
        
        return retrieved_docs

# Example Usage
if __name__ == "__main__":
    # 임베더 객체 생성
    embedder = Embedder(model_name="llama3.1", embedding_dimension=768, device_id=0)
    
    # JSON 데이터 처리
    json_data = [
        {"title": "뉴욕증시 기업 실적 소화하며 숨 고르기", "content": "미국 뉴욕증시의 3대 지수가..."},
        {"title": "삼성전자 주가 상승", "content": "삼성전자가 10% 상승했다..."}
    ]
    
    # 데이터 처리: 하나씩 처리하여 FAISS 인덱스에 추가
    for item in json_data:
        embedder.add_to_faiss(item)

    # 질문을 통해 유사한 문서 검색
    query = "뉴욕 증시 상황"
    result = embedder.query_faiss(query, top_k=1)

    # 검색된 결과 출력
    if result:
        print(f"유사한 문서 제목: {result[0]['title']}")
    else:
        print("유사한 문서를 찾을 수 없습니다.")
