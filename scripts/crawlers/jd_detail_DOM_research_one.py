import csv
import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "jobkorea_pages1_10_requests_candidates.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "research" / "jd_detail"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_first_n_job_urls(csv_path: str, n=3):
    urls = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("text", "").strip()
            url = row.get("href", "").strip()

            if title and "/Recruit/GI_Read/" in url:
                urls.append((title, url))
                if len(urls) >= n:
                    break
    if not urls:
        raise ValueError("CSV에서 공고 URL을 찾지 못했습니다.")
    return urls


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# JobKorea는 Python의 기본 User-Agent를 차단하므로 브라우저 User-Agent 설정이 필요합니다.
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

targets = read_first_n_job_urls(str(INPUT_CSV), n=3)

for idx, (title, url) in enumerate(targets, 1):
    print(f"\n[TARGET {idx}]")
    print("title:", title)
    print("url:", url)

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to retrieve page {idx}: HTTP status {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # 전체 페이지 텍스트
        body_text = soup.body.get_text() if soup.body else soup.get_text()
        body_text = clean_text(body_text)

        # 전체 HTML
        html = response.text

        # 모든 h태그
        headings = []
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(strip=True)
            if text:
                headings.append({
                    "tag": h.name.upper(),
                    "text": text,
                    "className": " ".join(h.get("class", [])) if h.get("class") else ""
                })

        # 전체 링크
        links = []
        for a in soup.find_all("a"):
            text = a.get_text(strip=True)
            href = a.get("href", "")
            if text or href:
                links.append({
                    "text": text,
                    "href": href,
                    "className": " ".join(a.get("class", [])) if a.get("class") else ""
                })

        # div/section/article 중 텍스트가 긴 후보 블록
        blocks = []
        elements = soup.find_all(["section", "article", "div"])
        for e_idx, e in enumerate(elements):
            text_val = e.get_text(" ", strip=True)
            text_length = len(text_val)
            if text_length > 300:
                blocks.append({
                    "index": e_idx,
                    "tag": e.name.upper(),
                    "className": " ".join(e.get("class", [])) if e.get("class") else "",
                    "id": e.get("id", ""),
                    "textLength": text_length,
                    "textPreview": text_val[:1000],
                    "outerHTMLPreview": str(e)[:1500]
                })
        
        # Sort and slice
        blocks.sort(key=lambda x: x["textLength"], reverse=True)
        blocks = blocks[:30]

        # 저장 대상 폴더 생성
        target_dir = OUTPUT_DIR / f"test_{idx}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 저장
        (target_dir / "detail_body_text.txt").write_text(body_text, encoding="utf-8")
        (target_dir / "detail_full_html.html").write_text(html, encoding="utf-8")

        with open(target_dir / "detail_headings.json", "w", encoding="utf-8") as f:
            json.dump(headings, f, ensure_ascii=False, indent=2)

        with open(target_dir / "detail_links.json", "w", encoding="utf-8") as f:
            json.dump(links, f, ensure_ascii=False, indent=2)

        with open(target_dir / "detail_text_blocks.json", "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        print(f"[DONE {idx}]")
        print("저장 폴더:", target_dir.resolve())
        print("- detail_body_text.txt")
        print("- detail_full_html.html")
        print("- detail_headings.json")
        print("- detail_links.json")
        print("- detail_text_blocks.json")

        print(f"[TEXT PREVIEW {idx}]")
        print(body_text[:500])
        print("="*50)

        # 차단 방지 딜레이
        time.sleep(1)

    except Exception as e:
        print(f"Error crawling detail page {idx}: {e}")
