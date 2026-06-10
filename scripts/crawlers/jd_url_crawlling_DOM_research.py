# 1~10페이지 DOM 구조 확인해서 JD 링크 후보 조사

from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "jobkorea_pages1_10_link_candidates.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

all_links = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    for page_no in range(1, 11):
        url = f"https://www.jobkorea.co.kr/Search/?stext=ai&FeatureCode=IDS&tabType=recruit&careerType=1&Page_No={page_no}"
        print(f"Crawling page {page_no}...")
        
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # 공고 상세 URL 후보만 수집
            page_links = page.locator("a[data-sentry-component='Title'][href*='/Recruit/GI_Read/']").evaluate_all("""
                els => els.map(a => {
                    const card = a.closest("article, li, div");
                    return {
                        text: a.innerText.trim(),
                        href: a.href,
                        className: a.className,
                        parentText: card ? card.innerText.slice(0, 500) : "",
                        outerHTML: a.outerHTML
                    }
                })
            """)
            all_links.extend(page_links)
            print(f"Page {page_no} crawled: found {len(page_links)} links.")
        except Exception as e:
            print(f"Error crawling page {page_no}: {e}")

    browser.close()

df = pd.DataFrame(all_links)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(df[["text", "href", "className"]].head(30))
print("후보 개수:", len(df))
print("저장 파일:", OUTPUT_CSV)

