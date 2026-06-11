# =============================================================================
# 1_build_clean_corpus.py  ―  잡코리아 JD 정제 코퍼스 빌더 (파이프라인 1단계)
# =============================================================================
# [이 코드가 하는 일]
#   data/raw/ 안에 있는 잡코리아 채용공고 CSV 파일들을 "모두" 읽어서,
#   검색/임베딩 학습에 바로 쓸 수 있는 '깨끗한 코퍼스'로 가공합니다.
#
#   처리 순서:
#     1) data/raw/ 의 jobkorea_jd_details_requests*.csv 파일 전체 로드
#     2) 채용공고 본문(iframe_jd_text)만 추출 (빈 값 제거)
#     3) 텍스트 정규화: 불릿 기호 통일 → NFKC 정규화 → 공백 정리
#     4) 보일러플레이트 제거: "면접일정 추후 통보", 채용절차법 조문 등
#        직무와 무관한 정형 문구 라인을 걷어냄
#     5) 중복 공고 제거
#
#   산출물 2개:
#     - data/processed/jd_corpus.txt    : 1줄 = 1공고. 토크나이저/MLM 학습용
#     - data/processed/jd_clean.parquet : company/url 메타 포함. 인덱싱/평가셋용
#
# [왜 1단계인가]
#   토크나이저 학습이든 KoSPLADE 인덱싱이든 도메인 적응(DAPT)이든,
#   모든 후속 작업이 이 정제 코퍼스를 공통 입력으로 사용합니다.
# =============================================================================

import glob
import os
import re
import unicodedata
import pandas as pd

# -----------------------------------------------------------------------------
# 경로 설정
#   - RAW_GLOB : data/raw 안의 모든 잡코리아 CSV (파일이 몇 개든 전부 매칭)
#   - OUT_DIR  : 정제 결과를 저장할 폴더 (없으면 자동 생성)
# -----------------------------------------------------------------------------
RAW_GLOB = "data/raw/jobkorea_jd_details_requests*.csv"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 보일러플레이트 정의
#   실측상 수만 건의 공고에 똑같이 반복되는 정형 문구들이다.
#   직무 신호가 아니라 "노이즈"이므로 학습 전에 제거해야 단어장이 깨끗해진다.
# -----------------------------------------------------------------------------
# (A) 라인 전체가 아래 문구와 '정확히' 일치하면 제거
BOILER_EXACT = {
    "포지션 및 자격요건", "모집분야 및 자격요건",
    "면접일정은 추후 통보됩니다.",
    "허위사실이 발견될 경우 채용이 취소될 수 있습니다.",
    "상세내용을 입력하세요",
    "학력 : 학력무관", "경력 : 경력무관",
    "서류전형 > 1차면접 > 2차면접 > 임원면접 > 최종합격",
}
# (B) 아래 패턴에 걸리는 라인 제거 (법령 조문 등 가변 텍스트)
BOILER_PATTERNS = [
    re.compile(r"^제\s*\d+\s*조\s*\(채용서류"),   # 채용절차법 제11조(채용서류의 반환) 등
    re.compile(r"채용서류의\s*반환"),
    re.compile(r"^[①-⑮]\s"),                     # 법령 '항' 번호(①②③…)로 시작하는 줄
    re.compile(r"대통령령으로\s*정하"),
    re.compile(r"^\(\s*[\d○]+\s*명\s*\)$"),       # ( 1명 ), ( 0명 ), ( ○명 )
]

# 불릿/가운뎃점 기호 통일용 (NFKC가 ㆍ를 옛한글 결합문자로 바꾸는 문제 예방)
BULLET_CHARS = re.compile(r"[ㆍᆞ·•∙‧・]")


def clean_text(t: str) -> str:
    """공고 본문 한 건을 정규화하고 보일러플레이트 라인을 제거한다."""
    # 1) 불릿 기호를 먼저 공백으로 치환 (NFKC 이전에 처리해야 깨지지 않음)
    t = BULLET_CHARS.sub(" ", str(t))
    # 2) 유니코드 NFKC 정규화 + non-breaking space(\xa0) 제거
    t = unicodedata.normalize("NFKC", t).replace("\xa0", " ")

    out_lines = []
    for ln in t.split("\n"):
        # 3) 연속 공백/탭을 한 칸으로 줄이고 양끝 공백 제거
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        # 4) 빈 줄 또는 '정확히 일치'하는 보일러플레이트 제거
        if not ln or ln in BOILER_EXACT:
            continue
        # 5) 패턴성 보일러플레이트(법령 등) 제거
        if any(p.search(ln) for p in BOILER_PATTERNS):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def main():
    # -------------------------------------------------------------------------
    # [1] data/raw 안의 CSV 파일을 '모두' 로드
    #     glob 패턴이 jobkorea_jd_details_requests* 로 시작하는 파일 전부를 잡는다.
    # -------------------------------------------------------------------------
    files = sorted(glob.glob(RAW_GLOB))
    print(f"[load] data/raw 에서 CSV {len(files)}개 발견")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"  ! 건너뜀 {os.path.basename(fp)}: {e}")
            continue
        # 본문 컬럼이 없는 파일은 스킵
        if "iframe_jd_text" not in df.columns:
            continue
        # 필요한 컬럼만 유지 (있는 것만)
        keep = [c for c in ("company", "url", "iframe_jd_text") if c in df.columns]
        frames.append(df[keep])

    # 모든 파일을 하나의 표로 합치기
    raw = pd.concat(frames, ignore_index=True)
    print(f"[rows] 전체 행 수 = {len(raw):,}")

    # -------------------------------------------------------------------------
    # [2] 본문이 비어있는(NaN) 행 제거 후, 각 공고를 정제
    # -------------------------------------------------------------------------
    raw = raw.dropna(subset=["iframe_jd_text"]).copy()
    raw["clean_text"] = raw["iframe_jd_text"].map(clean_text)
    # 정제 후 너무 짧아진(=거의 보일러플레이트뿐이던) 잔여물 제거
    raw = raw[raw["clean_text"].str.len() >= 30]

    # -------------------------------------------------------------------------
    # [3] 중복 공고 제거 (정제된 본문 기준)
    # -------------------------------------------------------------------------
    before = len(raw)
    raw = raw.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    print(f"[dedup] {before:,} → {len(raw):,} 고유 공고")

    # -------------------------------------------------------------------------
    # [4] 산출물 1: 학습용 코퍼스 (1줄 = 1공고, 내부 줄바꿈은 공백으로 평탄화)
    # -------------------------------------------------------------------------
    corpus_path = os.path.join(OUT_DIR, "jd_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for t in raw["clean_text"]:
            f.write(t.replace("\n", " ") + "\n")

    # -------------------------------------------------------------------------
    # [5] 산출물 2: 메타 포함 정제본 (인덱싱/평가셋 제작용)
    #     parquet 우선, 환경에 엔진이 없으면 CSV로 폴백
    # -------------------------------------------------------------------------
    out2 = os.path.join(OUT_DIR, "jd_clean.parquet")
    try:
        raw.to_parquet(out2, index=False)
    except Exception:
        out2 = os.path.join(OUT_DIR, "jd_clean.csv")
        raw.to_csv(out2, index=False)

    # -------------------------------------------------------------------------
    # [6] 최종 통계 출력
    # -------------------------------------------------------------------------
    chars = int(raw["clean_text"].str.len().sum())
    print(f"[done] 코퍼스 → {corpus_path}")
    print(f"[done] 정제본 → {out2}")
    print(f"[stats] 고유 공고 {len(raw):,}건 · 총 {chars:,}자")


if __name__ == "__main__":
    main()

