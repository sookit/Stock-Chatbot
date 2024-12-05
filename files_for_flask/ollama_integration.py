from ollama import Client
import torch
import time  # 응답 시간 측정을 위한 time 모듈

class OllamaLLM:
    def __init__(self, model_name="mistral-small"):
        # Ollama 클라이언트 인스턴스 생성
        self.client = Client()
        self.model_name = model_name

        # 모델 이름이 'mistral-small'인 경우 로컬에서 해당 모델을 사용 중임을 알림
        if self.model_name == "mistral-small":
            print("로컬 mistral-small 사용중")

        # GPU 사용 여부 확인
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU 사용 불가, CPU에서 실행 중")

            

    def query(self, prompt, streaming=False):
        # 메시지 형식 준비
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()  # 시작 시간 기록

        if streaming:
            # 스트리밍 응답 처리
            response = self.client.chat(model=self.model_name, messages=messages, stream=True)
            full_response = ""
            for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                full_response += content
                print(content, end="", flush=True)  # 스트리밍되는 응답을 실시간으로 출력
            
            end_time = time.time()  # 종료 시간 기록
            print(f"\n응답 시간: {end_time - start_time:.2f}초")  # 응답 시간 출력
            return full_response
        else:
            # 스트리밍이 아닌 일반 응답 처리
            response = self.client.chat(model=self.model_name, messages=messages)
            
            end_time = time.time()  # 종료 시간 기록
            print(f"응답 시간: {end_time - start_time:.2f}초")  # 응답 시간 출력
            
            return response.get("message", {}).get("content", "No response")