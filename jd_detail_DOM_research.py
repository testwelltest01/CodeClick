from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import csv
import re
import time

INPUT_CSV = "jobkorea_page1_link_candidates.csv"
OUTPUT_CSV = "jobkorea_jd_details.csv"

# 1. 입력 CSV에서 공고 URL 리스트 읽기
jobs = []
seen = set()

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        title = row.get("text", "").strip()
        url = row.get("href", "").strip()

        if not title:
            continue

        if "/Recruit/GI_Read/" not in url:
            continue

        if url in seen:
            continue

        seen.add(url)

        jobs.append({
            "list_title": title,
            "url": url
        })

print(f"[INFO] 수집 대상 공고 수: {len(jobs)}")


# 2. 출력 CSV 준비
fieldnames = [
    "company",
    "url",
    "iframe_jd_text",
    "main_jd_text",
    "json_ld",
    "status",
    "error",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=80
        )

        page = browser.new_page(
            user_agent=(
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        for idx, job in enumerate(jobs, start=1):
            print(f"\n[{idx}/{len(jobs)}] {job['list_title']}")
            print(job["url"])

            try:
                page.goto(job["url"], wait_until="domcontentloaded", timeout=60000)

                try:
                    page.wait_for_load_state("load", timeout=15000)
                except Exception:
                    pass

                page.wait_for_timeout(2000)
                
                # 회사명
                company = ""
                if page.locator("h2").count() > 0:
                    company = page.locator("h2").first.inner_text().strip()

                # JSON-LD
                json_ld = ""
                json_scripts = page.locator("script[type='application/ld+json']").evaluate_all("""
                    els => els.map(e => e.innerText)
                """)
                json_ld = "\n".join(json_scripts).strip()


                # iframe 안 상세 JD 텍스트 수집
                iframe_jd_text = ""

                try:
                    iframe_src = ""

                    if page.locator("iframe[title='상세 모집 요강']").count() > 0:
                        iframe_src = page.locator("iframe[title='상세 모집 요강']").first.get_attribute("src")

                    if iframe_src:
                        iframe_url = urljoin(job["url"], iframe_src)

                        iframe_page = browser.new_page(
                            user_agent=(
                                "Chrome/120.0.0.0 Safari/537.36"
                            )
                        )
                        
                        iframe_page.goto(iframe_url, wait_until="domcontentloaded", timeout=60000)

                        try:
                            iframe_page.wait_for_load_state("load", timeout=15000)
                        except Exception:
                            pass

                        iframe_page.wait_for_timeout(3000)

                        # iframe 안 텍스트
                        try:
                            iframe_jd_text = iframe_page.locator("body").inner_text(timeout=5000).strip()
                        except Exception:
                            iframe_jd_text = ""

                        iframe_page.close()

                except Exception as e:
                    print("[IFRAME FAIL]", e)


                # 바깥 정형 JD 영역
                main_parts = []

                selectors = [
                    "[data-sentry-component='RecruitmentGuidelines']",
                    "[data-sentry-component='Qualification']",
                    "#application-section",
                    "#company-section",
                    "[data-sentry-component='BenefitCard']",
                ]

                for selector in selectors:
                    try:
                        if page.locator(selector).count() > 0:
                            text = page.locator(selector).first.inner_text(timeout=5000)
                            main_parts.append(text)
                    except Exception:
                        pass

                main_jd_text = "\n\n".join(main_parts)

                # 텍스트 정리
                iframe_jd_text = re.sub(r"\n{3,}", "\n\n", iframe_jd_text).strip()
                main_jd_text = re.sub(r"\n{3,}", "\n\n", main_jd_text).strip()

                row = {
                    "company": company,
                    "url": job["url"],
                    "iframe_jd_text": iframe_jd_text,
                    "main_jd_text": main_jd_text,
                    "json_ld": json_ld,
                    "status": "success",
                    "error": "",
                }

                writer.writerow(row)
                f.flush()

                print("[OK]", company)
                print("iframe 길이:", len(iframe_jd_text))
                print("main 길이:", len(main_jd_text))

            except Exception as e:
                row = {
                    "company": "",
                    "url": job["url"],
                    "iframe_jd_text": "",
                    "main_jd_text": "",
                    "json_ld": "",
                    "status": "fail",
                    "error": str(e),
                }

                writer.writerow(row)
                f.flush()

                print("[FAIL]", e)

            time.sleep(1.5)

        browser.close()

print(f"\n[DONE] 저장 완료: {OUTPUT_CSV}")
