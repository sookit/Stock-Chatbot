import requests

def test_query_and_time_limit():
    # 사용자로부터 쿼리와 시간 제한 입력 받기
    query = input("Enter your query for Naver News crawling: ")
    time_limit = input("Enter the time limit for crawling (in seconds): ")

    # API에 쿼리와 시간 제한을 동시에 전송
    response = requests.post('http://192.168.1.101:5000/api/query', json={"query": query, "time_limit": int(time_limit)})
    
    if response.status_code == 200:
        print(f"Server response: {response.json()}")
    else:
        print(f"Error: {response.json()}")

def test_chat():
    # 사용자로부터 분석 쿼리 입력 받기
    analysis_query = input("Enter your analysis query: ")
    
    # API에 분석 쿼리 전송
    response = requests.post('http://192.168.1.101:5000/api/chat', json={"analysis_query": analysis_query})
    
    if response.status_code == 200:
        print(f"Server response: {response.json()}")
    else:
        print(f"Error: {response.json()}")

if __name__ == '__main__':
    print("Testing /api/query endpoint with query and time limit...")
    test_query_and_time_limit()  # 쿼리와 시간 제한 입력받기

    print("\nTesting /api/chat endpoint...")
    test_chat()  # 분석 쿼리 입력받기
