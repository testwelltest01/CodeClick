# 1페이지만 DOM 구조 확인해서 JD 링크 후보 조사

from playwright.sync_api import sync_playwright
import pandas as pd

URL = "https://www.jobkorea.co.kr/Search?stext=ai+engineer&tabType=recruit&careerType=1&edu=0&jobtype=1&Page_No=1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto(URL, wait_until="networkidle", timeout=60000)

    # 공고 상세 URL 후보만 수집
    links = page.locator("a[data-sentry-component='Title'][href*='/Recruit/GI_Read/']").evaluate_all("""
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

    browser.close()

df = pd.DataFrame(links)
df.to_csv("jobkorea_page1_link_candidates.csv", index=False, encoding="utf-8-sig")

print(df[["text", "href", "className"]].head(30))
print("후보 개수:", len(df))