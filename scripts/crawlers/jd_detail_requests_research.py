import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import csv
import re
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "jobkorea_jd_details_requests.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 임시 데이터로 직접 지정된 범위의 공고 URL 생성
jobs = []
for gi_no in range(49100000, 49300000):
    jobs.append({
        "list_title": f"임시 공고 ({gi_no})",
        "url": f"https://www.jobkorea.co.kr/Recruit/GI_Read/{gi_no}"
    })

print(f"[INFO] 수집 대상 공고 수 (임시 범위): {len(jobs)}")


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

# JobKorea 기본 User-Agent 차단 방지
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

def get_output_path(base_path: Path, part: int) -> Path:
    return base_path.parent / f"{base_path.stem}_{part}{base_path.suffix}"

f = None
writer = None
part_no = 0

try:
    for idx, job in enumerate(jobs, start=1):
        # 1000개 단위로 새로운 파일 생성
        if (idx - 1) % 1000 == 0:
            if f:
                f.close()
            part_no += 1
            current_output_csv = get_output_path(OUTPUT_CSV, part_no)
            f = open(current_output_csv, "w", newline="", encoding="utf-8-sig")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            print(f"\n[INFO] 새 파일 시작: {current_output_csv.name}")

        print(f"\n[{idx}/{len(jobs)}] {job['list_title']}")
        print(job["url"])

        try:
            response = requests.get(job["url"], headers=headers, timeout=15)
            if response.status_code != 200:
                raise Exception(f"HTTP status code {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")

            # 회사명
            company = ""
            h2_tag = soup.find("h2")
            if h2_tag:
                company = h2_tag.get_text(strip=True)

            # JSON-LD
            json_scripts = soup.find_all("script", type="application/ld+json")
            json_ld = "\n".join([s.get_text().strip() for s in json_scripts]).strip()

            # iframe 안 상세 JD 텍스트 수집
            iframe_jd_text = ""
            try:
                # Playwright와 달리 requests 방식(SSR)에서는 JS가 실행되지 않아 iframe 태그가 HTML에 생성되지 않습니다.
                # 대신, Next.js의 state 구조나 URL 규칙을 이용해 iframe 주소를 직접 생성해 요청합니다.
                gno_match = re.search(r"/GI_Read/(\d+)", job["url"])
                oem_match = re.search(r"Oem_Code=([^&]+)", job["url"])
                
                if gno_match:
                    gno = gno_match.group(1)
                    oem_code = oem_match.group(1) if oem_match else "C1"
                    iframe_url = f"https://www.jobkorea.co.kr/Recruit/GI_Read_Comt_Ifrm?Oem_Code={oem_code}&Gno={gno}&isHiringCenter=false&hideMapView=false"
                    
                    iframe_res = requests.get(iframe_url, headers=headers, timeout=15)
                    if iframe_res.status_code == 200:
                        iframe_soup = BeautifulSoup(iframe_res.text, "html.parser")
                        iframe_body = iframe_soup.body if iframe_soup.body else iframe_soup
                        iframe_jd_text = iframe_body.get_text("\n", strip=True)
                    else:
                        print(f"  [WARNING] Failed to fetch iframe, status code: {iframe_res.status_code}")
                else:
                    print("  [WARNING] Could not extract Gno (GI_No) from URL.")
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
                    el = soup.select_one(selector)
                    if el:
                        text = el.get_text("\n", strip=True)
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

        # 서버 과부하 방지 및 차단 예방용 딜레이
        time.sleep(1.0)
finally:
    if f:
        f.close()

print(f"\n[DONE] 저장 완료 (총 {part_no}개 분할 파일 저장됨)")
