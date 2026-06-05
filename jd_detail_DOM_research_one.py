from playwright.sync_api import sync_playwright
import csv
import json
import re
from pathlib import Path

INPUT_CSV = "jobkorea_page1_link_candidates.csv"
OUTPUT_DIR = Path("jd_detail_research")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_first_job_url(csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("text", "").strip()
            url = row.get("href", "").strip()

            if title and "/Recruit/GI_Read/" in url:
                return title, url

    raise ValueError("CSV에서 공고 URL을 찾지 못했습니다.")


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


title, url = read_first_job_url(INPUT_CSV)

print("[TARGET]")
print("title:", title)
print("url:", url)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=100)
    page = browser.new_page(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    )

    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(3000)

    # 전체 페이지 텍스트
    body_text = page.locator("body").inner_text(timeout=30000)
    body_text = clean_text(body_text)

    # 전체 HTML
    html = page.content()

    # 모든 h태그
    headings = page.locator("h1, h2, h3, h4").evaluate_all("""
        els => els.map(e => ({
            tag: e.tagName,
            text: e.innerText.trim(),
            className: e.className
        })).filter(x => x.text.length > 0)
    """)

    # 전체 링크
    links = page.locator("a").evaluate_all("""
        els => els.map(a => ({
            text: a.innerText.trim(),
            href: a.href,
            className: a.className
        })).filter(x => x.text.length > 0 || x.href.length > 0)
    """)

    # div/section/article 중 텍스트가 긴 후보 블록
    blocks = page.locator("section, article, div").evaluate_all("""
        els => els.map((e, idx) => ({
            index: idx,
            tag: e.tagName,
            className: e.className,
            id: e.id,
            textLength: e.innerText ? e.innerText.trim().length : 0,
            textPreview: e.innerText ? e.innerText.trim().slice(0, 1000) : "",
            outerHTMLPreview: e.outerHTML.slice(0, 1500)
        }))
        .filter(x => x.textLength > 300)
        .sort((a, b) => b.textLength - a.textLength)
        .slice(0, 30)
    """)

    browser.close()


# 저장
(OUTPUT_DIR / "detail_body_text.txt").write_text(body_text, encoding="utf-8")
(OUTPUT_DIR / "detail_full_html.html").write_text(html, encoding="utf-8")

with open(OUTPUT_DIR / "detail_headings.json", "w", encoding="utf-8") as f:
    json.dump(headings, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "detail_links.json", "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "detail_text_blocks.json", "w", encoding="utf-8") as f:
    json.dump(blocks, f, ensure_ascii=False, indent=2)

print("\n[DONE]")
print("저장 폴더:", OUTPUT_DIR.resolve())
print("- detail_body_text.txt")
print("- detail_full_html.html")
print("- detail_headings.json")
print("- detail_links.json")
print("- detail_text_blocks.json")

print("\n[TEXT PREVIEW]")
print(body_text[:2000])