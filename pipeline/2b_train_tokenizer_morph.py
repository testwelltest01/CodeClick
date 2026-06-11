# =============================================================================
# 2b_train_tokenizer_morph.py  ―  형태소 분석 기반 JD 토크나이저 (파이프라인 2단계 변형)
# =============================================================================
# [이 코드가 하는 일]
#   2단계(2_train_tokenizer.py)는 정제 코퍼스에 WordPiece를 '바로' 학습했다.
#   이 변형판은 그 전에 **형태소 분석기(Kiwi)로 어절을 먼저 쪼갠 뒤** WordPiece를
#   학습한다. 한국어는 교착어라 "허위사실 / 허위사실이 / 허위사실을"처럼 같은
#   어근에 조사가 붙은 표면형이 각각 vocab 슬롯을 잡아먹는데, 형태소로 미리
#   분리하면 어근 "허위사실" 하나 + 조사 "이/을"을 공용 토큰으로 처리할 수 있다.
#   → 같은 vocab 예산으로 더 많은 고유 어근을 담고, 검색 매칭도 좋아진다.
#   (근거: KR-BERT(arXiv:2008.03979), 한국어 형태소 토크나이즈 연구 arXiv:2311.03928)
#
#   처리 순서:
#     1) 1단계 산출물 data/processed/jd_corpus.txt 로드
#     2) Kiwi 형태소 분석기로 각 줄을 형태소 단위로 분해 → jd_corpus_morph.txt 저장
#        (도메인 용어 과분할 방지를 위해 사용자 사전 등록: 4대보험, B2B 등)
#     3) 형태소 코퍼스로 WordPiece 학습
#     4) models/jd_tokenizer_morph/ 에 저장 + 품질 점검(샘플/OOV/중복어근 비교)
#
# [설치]
#     pip install kiwipiepy tokenizers
#     (Kiwi는 시스템 설치 없이 pip만으로 동작. Mecab 대비 설치가 간단)
#
# [실행]
#     python pipeline/2b_train_tokenizer_morph.py
#
# [주의]
#   이 토크나이저도 그 자체로는 sparse 검색 모델이 아니다(2단계와 동일).
#   형태소 분리는 검색 정확도엔 유리하나 복합어/약어를 과분할할 수 있으므로,
#   실제 채택은 2단계(raw)와 평가셋으로 비교한 뒤 결정한다.
# =============================================================================

import os
import sys
from tokenizers import BertWordPieceTokenizer

# -----------------------------------------------------------------------------
# 설정값 (2단계와 동일한 의미. 산출 경로만 _morph 로 분리해 raw 버전과 공존)
# -----------------------------------------------------------------------------
CORPUS_PATH = "data/processed/jd_corpus.txt"            # 입력: 1단계 정제 코퍼스
MORPH_CORPUS_PATH = "data/processed/jd_corpus_morph.txt"  # 중간 산출: 형태소 분해본
OUT_DIR = "models/jd_tokenizer_morph"                  # 출력: 형태소 기반 토크나이저
VOCAB_SIZE = 32000
MIN_FREQ = 3
LOWERCASE = False
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# 형태소 분석기가 통째로 유지해야 할 도메인 용어 (과분할 방지용 사용자 사전).
# 예: "4대보험"을 "4 / 대 / 보험"으로 쪼개지 않도록 하나의 고유명사로 등록.
USER_WORDS = [
    "4대보험", "주5일", "연차수당", "퇴직연금", "포트폴리오",
    "백엔드", "프론트엔드", "데이터엔지니어", "풀스택",
    "B2B", "B2C", "CRM", "ERP", "KPI", "SQL", "AWS",
]

# 토큰화 품질 확인용 샘플
SAMPLE_TEXTS = [
    "백엔드 개발자를 채용합니다. Python, Django, AWS 경험 우대.",
    "B2B 영업 및 신규 거래처 발굴, 매출 관리 담당. 경력 3년 이상.",
    "간호조무사 모집 / 4대보험 / 주 5일 / 인근 거주자 우대.",
]


def build_morph_corpus():
    """1단계 코퍼스를 형태소 단위로 분해해 MORPH_CORPUS_PATH 로 저장한다."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        sys.exit("[중단] kiwipiepy 미설치.  pip install kiwipiepy 후 다시 실행하세요.")

    if not os.path.exists(CORPUS_PATH):
        sys.exit(
            f"[중단] 입력 코퍼스가 없습니다: {CORPUS_PATH}\n"
            f"        먼저 1단계를 실행하세요:  python pipeline/1_build_clean_corpus.py"
        )

    # ★ 속도 핵심: 병렬 워커 수는 Kiwi 생성자에 지정한다(kiwipiepy 0.23 기준).
    #   이후 tokenize()에 '리스트 전체'를 넘기면 입력 순서를 유지한 채 여러 코어로
    #   병렬 처리한다. 단건 루프보다 수~수십 배 빠름.
    n_workers = min(8, os.cpu_count() or 1)
    kiwi = Kiwi(num_workers=n_workers)

    # 도메인 용어를 고유명사(NNP)로 등록해 과분할 방지 (예: "4대보험"을 한 토큰으로 유지)
    for w in USER_WORDS:
        kiwi.add_user_word(w, "NNP")

    with open(CORPUS_PATH, encoding="utf-8") as fin:
        lines = [ln.strip() for ln in fin if ln.strip()]
    print(f"[morph] 형태소 분해 시작 · 입력 {len(lines):,} 공고 (Kiwi, 워커 {n_workers}개)")

    # 각 토큰의 '표면형(form)'만 사용(품사 태그는 학습에 불필요), 공백 형태소는 제외.
    with open(MORPH_CORPUS_PATH, "w", encoding="utf-8") as fout:
        for i, toks in enumerate(kiwi.tokenize(lines)):
            forms = [t.form for t in toks if t.form.strip()]
            fout.write(" ".join(forms) + "\n")
            if (i + 1) % 20000 == 0:
                print(f"  · {i + 1:,} 공고 처리")
    print(f"[morph] 완료 → {MORPH_CORPUS_PATH}")


def train_tokenizer():
    """형태소 코퍼스로 WordPiece 토크나이저를 학습/저장하고 품질을 점검한다."""
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=LOWERCASE,
    )
    print(f"[train] vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQ}, lowercase={LOWERCASE}")
    tokenizer.train(
        files=[MORPH_CORPUS_PATH],   # ★ 차이점: 형태소로 분해된 코퍼스를 학습
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        wordpieces_prefix="##",
    )
    tokenizer.save_model(OUT_DIR)
    print(f"[save] 토크나이저 저장 → {OUT_DIR}/  (실제 vocab={tokenizer.get_vocab_size():,})")

    # ---- 품질 점검 (a) 샘플 토큰화 ----
    print("\n[check] 샘플 토큰화 결과")
    for s in SAMPLE_TEXTS:
        print(f"  · {s}")
        print(f"    → {tokenizer.encode(s).tokens}\n")

    # ---- 품질 점검 (b) OOV([UNK]) 비율 ----
    print("[check] OOV([UNK]) 비율 (형태소 코퍼스 앞 5천 줄)")
    total, unk = 0, 0
    with open(MORPH_CORPUS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            toks = tokenizer.encode(line.strip()).tokens
            total += len(toks)
            unk += sum(1 for t in toks if t == "[UNK]")
    if total:
        print(f"  · 토큰 {total:,}개 중 [UNK] {unk:,}개 = {unk / total * 100:.3f}%")

    # ---- 품질 점검 (c) 어근 중복 여부 (형태소 분리 효과 확인) ----
    vocab = set(tokenizer.get_vocab().keys())
    print("\n[check] 어근/표면형 중복 점검 (형태소 분리가 잘 됐다면 어근만 남아야 함)")
    for w in ["허위사실", "허위사실이", "허위사실을", "개발", "개발자", "개발자를"]:
        print(f"  · {w!r} in vocab: {w in vocab}")


def main():
    build_morph_corpus()   # 1) 형태소 코퍼스 생성
    train_tokenizer()      # 2) WordPiece 학습 + 점검
    print("\n[done] 2b단계 완료. 2단계(raw)와 평가셋으로 비교 후 채택을 결정하세요.")


if __name__ == "__main__":
    main()
