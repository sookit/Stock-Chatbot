import os
import urllib.request
import json
import time
from newspaper import Article

# Naver API credentials
client_id = "_GOqrACtWl33_XNAlMhq"
client_secret = "FqGHPmU_CJ"

def extract_article_text_and_images(url):
    """기사 본문을 추출하는 함수"""
    clean_url = url.replace("\\", "")  # 역슬래시 제거
    article = Article(clean_url)
    article.download()
    article.parse()
    
    # 기사 본문 텍스트
    article_text = article.text
    return article_text

def search_naver_news(query, start=1, display=100):
    """네이버 검색 API를 사용해 뉴스를 검색하는 함수"""
    encText = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display={display}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        return json.loads(response_body.decode('utf-8'))
    else:
        print("Error Code:", rescode)
        return None

def get_articles_info_with_time_limit(query, max_time=10, display=100):
    """설정된 시간 동안 뉴스를 가져오는 함수"""
    articles_info = []
    article_id = 1
    
    start_time = time.time()  # 시작 시간 기록
    
    for start in range(1, 1001, display):  # 최대 1000개까지 시도
        # 현재 시간이 제한 시간을 초과하면 중단
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached.")
            break
        
        # 네이버 뉴스 검색
        news_data = search_naver_news(query, start=start, display=display)
        if not news_data:
            continue
        
        for item in news_data.get('items', []):
            title = item['title']
            original_link = item['originallink']
            pub_date = item.get('pubDate', '')  # 게시 날짜
            
            try:
                article_text = extract_article_text_and_images(original_link)
            except Exception as e:
                article_text = f"Failed to extract article. Error: {e}"
            
            articles_info.append({
                'id': article_id,
                'keyword': query,
                'title': title,
                'content': article_text,
                'link': original_link,
                'pub_date': pub_date
            })
            article_id += 1
        
        # 현재 시간이 제한 시간을 초과하면 중단
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached.")
            break
    
    return articles_info

def save_articles_to_json(articles_info, query):
    """추출한 기사를 JSON 파일로 저장하는 함수"""
    # 현재 파일 기준으로 프로젝트 루트 디렉터리 탐지
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 부모 디렉터리 이동
    
    # 저장 폴더 경로 설정
    folder_path = os.path.join(project_root, "data", "articles_data")
    
    # 폴더가 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 저장할 파일의 번호 설정
    file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]) + 1
    filename = f"articles_{file_count}.json"
    
    # 파일 경로
    file_path = os.path.join(folder_path, filename)
    
    # JSON 파일로 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(articles_info, f, ensure_ascii=False, indent=4)
    
    print(f"Articles saved to {file_path}")

# 단독 실행 시 기본 쿼리 실행
if __name__ == "__main__":
    # 기본 쿼리 설정
    query = "ETF"  # 기본 쿼리
    time_limit = 10  # 10초 동안 크롤링

    print(f"Executing default query: {query} with a time limit of {time_limit} seconds...")
    
    # 뉴스 기사 크롤링
    articles_info = get_articles_info_with_time_limit(query, max_time=time_limit)

    # JSON 파일로 저장
    if articles_info:
        save_articles_to_json(articles_info, query)

    print(f"Total articles fetched: {len(articles_info)}")
