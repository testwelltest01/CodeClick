# 1~10페이지 DOM 구조 확인해서 JD 링크 후보 조사 (requests 방식)

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "jobkorea_pages1_10_requests_candidates.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# JobKorea는 Python의 기본 User-Agent를 차단하므로 브라우저 User-Agent 설정이 필요합니다.
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

all_links = []

for page_no in range(1, 11):
    url = f"https://www.jobkorea.co.kr/Search/?stext=ai&FeatureCode=IDS&tabType=recruit&careerType=1&Page_No={page_no}"
    print(f"Crawling page {page_no} with requests...")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page_no}: HTTP status {response.status_code}")
            continue
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # a[data-sentry-component='Title'] 요소들 검색
        a_tags = soup.find_all("a", attrs={"data-sentry-component": "Title"})
        
        page_links = []
        for a in a_tags:
            href = a.get("href", "")
            # '/Recruit/GI_Read/'를 포함하는 href 필터링
            if "/Recruit/GI_Read/" not in href:
                continue
                
            # 상대 경로를 절대 경로로 변환
            absolute_href = href
            if href.startswith("/"):
                absolute_href = f"https://www.jobkorea.co.kr{href}"
            elif not href.startswith("http"):
                absolute_href = f"https://www.jobkorea.co.kr/{href}"
                
            text = a.get_text(strip=True)
            class_name = " ".join(a.get("class", [])) if a.get("class") else ""
            
            # closest("article, li, div")와 유사하게 부모 노드 탐색
            card = None
            parent = a.parent
            while parent:
                if parent.name in ["article", "li", "div"]:
                    card = parent
                    break
                parent = parent.parent
                
            parent_text = card.get_text(" ", strip=True)[:500] if card else ""
            outer_html = str(a)
            
            page_links.append({
                "text": text,
                "href": absolute_href,
                "className": class_name,
                "parentText": parent_text,
                "outerHTML": outer_html
            })
            
        all_links.extend(page_links)
        print(f"Page {page_no} crawled with requests: found {len(page_links)} links.")
        
        # 서버 과부하 방지 및 차단 방지를 위한 약간의 딜레이
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error crawling page {page_no}: {e}")

df = pd.DataFrame(all_links)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(df[["text", "href", "className"]].head(30))
print("후보 개수:", len(df))
print("저장 파일:", OUTPUT_CSV)
