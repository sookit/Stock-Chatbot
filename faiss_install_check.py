import faiss
print(f"Faiss version: {faiss.__version__}")
import torch
print(torch.cuda.is_available())  # True면 GPU가 활성화됨
print(torch.cuda.get_device_name(0))  # 사용 중인 GPU 이름 출력
from ollama import Client

# Instantiate the client
client = Client()

# Define the conversation messages
messages = [{"role": "user", "content": "hi"}]

try:
    response = client.chat(model="mistral-small", messages=messages)
    print(f"Response: {response}")
except Exception as e:
    print(f"An error occurred: {e}")
