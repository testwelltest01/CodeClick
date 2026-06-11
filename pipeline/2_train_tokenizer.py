# =============================================================================
# 2_train_tokenizer.py  ―  잡코리아 JD WordPiece 토크나이저 학습 (파이프라인 2단계)
# =============================================================================
# [이 코드가 하는 일]
#   1단계(1_build_clean_corpus.py)가 만든 정제 코퍼스(jd_corpus.txt)를 입력으로
#   직무(JD) 도메인 특화 WordPiece 토크나이저를 학습합니다.
#
#   처리 순서:
#     1) data/processed/jd_corpus.txt 존재 확인 (없으면 1단계 먼저 실행하라고 안내)
#     2) BertWordPieceTokenizer 학습 (도메인 어휘 중심 vocab 생성)
#     3) models/jd_tokenizer/ 에 저장 (vocab.txt)
#     4) 샘플 문장으로 토큰화 결과 + 어휘 통계를 출력해 품질 즉시 확인
#
# [데이터가 계속 늘어나는 상황 대응]
#   공고가 20만까지 계속 수집되는 중이므로,
#   새 CSV가 추가될 때마다  ① 1단계를 다시 돌려 jd_corpus.txt를 갱신하고
#   ② 이 2단계를 다시 실행하면 됩니다.
#   이 스크립트는 항상 "현재 시점의 최신 코퍼스 전체"를 다시 학습합니다.
#   (토크나이저는 증분 학습이 아니라 전체 재학습이 표준입니다.)
#
# [중요한 주의]
#   이 토크나이저만으로는 KoSPLADE 같은 'sparse 검색 모델'이 되지 않습니다.
#   sparse 표현은 토크나이저가 아니라 'MLM head를 가진 사전학습 모델'이 만듭니다.
#   따라서 이 단계의 산출물은 (a) 도메인 어휘 분석/탐색용,
#   (b) 이후 도메인 적응(continual pretraining) 시 vocab 후보 검토용입니다.
#   실제 sparse 모델은 후속 단계(기존 한국어 BERT 도메인 적응)에서 만듭니다.
#
# CORPUS란? "모델을 학습시키거나 언어를 연구하기 위해 특정 목적을 가지고 수집한 대규모 텍스트 데이터의 뭉치"
#
# =============================================================================

import os
import sys
from tokenizers import BertWordPieceTokenizer

# -----------------------------------------------------------------------------
# 설정값
#   - CORPUS_PATH : 1단계 산출물 (1줄=1공고)
#   - OUT_DIR     : 학습된 토크나이저 저장 폴더
#   - VOCAB_SIZE  : 어휘 크기. 정제 코퍼스(약 6~7천만 자) 규모에 맞춰 32K 권장
#                   (원래 코드의 35K도 가능하나, 정제 후엔 32K가 더 깔끔)
#   - MIN_FREQ    : 이 횟수 미만으로 등장한 서브워드는 vocab에서 제외 (노이즈 억제)
#   - LOWERCASE   : False 권장. True면 B2B/MD/OA 같은 직무 약어가 소문자화되어
#                   의미 구분이 약해짐. 한글은 대소문자가 없어 영향 없음.
# -----------------------------------------------------------------------------
CORPUS_PATH = "data/processed/jd_corpus.txt"
OUT_DIR = "models/jd_tokenizer"
VOCAB_SIZE = 32000
MIN_FREQ = 3
LOWERCASE = False

# 특수 토큰: 일반 단어가 아니라 모델이 구조적으로 쓰는 예약 토큰. 항상 vocab 맨 앞에 고정 배치된다.
#   [PAD]  : 배치 학습 시 짧은 문장을 같은 길이로 맞추는 채움(padding)용. attention에서 무시됨.
#   [UNK]  : vocab에 없는(=쪼개도 못 만드는) 글자를 대체하는 미등록(unknown) 토큰. 많을수록 나쁨.
#   [CLS]  : 문장 맨 앞에 붙는 분류(classification)용 요약 토큰. 문장 전체 임베딩 대표로 쓰임.
#   [SEP]  : 문장 끝/문장 경계 구분(separator) 토큰. 문장쌍 입력(질의-문서)에서 둘을 가름.
#   [MASK] : MLM 학습에서 가린 단어 자리를 나타내는 토큰. ★SPLADE/도메인 적응의 핵심 토큰.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# 토큰화 품질을 눈으로 확인할 샘플 문장 (직무 도메인 표현)
SAMPLE_TEXTS = [
    "백엔드 개발자를 채용합니다. Python, Django, AWS 경험 우대.",
    "B2B 영업 및 신규 거래처 발굴, 매출 관리 담당. 경력 3년 이상.",
    "간호조무사 모집 / 4대보험 / 주 5일 / 인근 거주자 우대.",
]


def main():
    # -------------------------------------------------------------------------
    # [1] 입력 코퍼스 확인 — 없으면 1단계를 먼저 실행하도록 안내 후 종료
    # -------------------------------------------------------------------------
    if not os.path.exists(CORPUS_PATH):
        sys.exit(
            f"[중단] corpus가 없습니다: {CORPUS_PATH}\n"
            f" 1단계를 먼저 실행하여 corpus를 생성하세요:  python pipeline/1_build_clean_corpus.py"
        )

    n_lines = sum(1 for _ in open(CORPUS_PATH, encoding="utf-8"))
    print(f"[load] 코퍼스 {CORPUS_PATH} · {n_lines:,} 공고")

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # [2] WordPiece 토크나이저 학습
    #     - strip_accents=False, handle_chinese_chars=False : 한국어/한자 혼용에 적합
    #     - clean_text=True : 제어문자 제거 등 기본 정리
    # -------------------------------------------------------------------------
    tokenizer = BertWordPieceTokenizer(
        # clean_text : 제어문자(\x00 등)와 비정상 공백을 제거하는 기본 텍스트 정리.
        #              True 권장. 크롤링 텍스트엔 보이지 않는 잡문자가 섞이기 쉽다.
        clean_text=True,
        # handle_chinese_chars : 한자(CJK)를 글자 단위로 강제 분리할지 여부.
        #              True면 "理事"가 "理"+"事"처럼 한 글자씩 쪼개진다.
        #              한국어 공고엔 한자가 단어처럼 쓰이므로(예: 有, 名) False로 둬서 통째 학습.
        handle_chinese_chars=False,
        # strip_accents : 발음 구별 기호(é→e 등)를 제거할지 여부.
        #              한글엔 무관하지만, True면 한글 자모 결합/외래어 표기가 깨질 수 있어 False.
        strip_accents=False,
        # lowercase : 영문 대문자를 소문자로 변환할지 여부 (위 LOWERCASE=False 사용).
        #              False면 B2B/MD/AWS 같은 직무 약어가 보존된다. 한글은 영향 없음.
        lowercase=LOWERCASE,
    )
    print(f"[train] vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQ}, lowercase={LOWERCASE}")
    tokenizer.train(
        # files : 학습에 사용할 텍스트 파일 목록. 여기선 1단계가 만든 정제 코퍼스 하나.
        #         (여러 파일을 줄 수도 있지만, 우리는 1단계에서 이미 하나로 합쳐 둠)
        files=[CORPUS_PATH],
        # vocab_size : 만들 어휘 사전의 '목표' 크기(특수 토큰 포함). 학습이 이 크기에 맞춰
        #              가장 빈번한 서브워드 조합을 골라 병합한다. 클수록 단어를 통째로
        #              담지만 희귀·노이즈 토큰도 늘고, 작을수록 잘게 쪼개진다.
        vocab_size=VOCAB_SIZE,
        # min_frequency : 이 횟수 미만 등장한 후보는 vocab에 넣지 않음(노이즈·오타 억제).
        #                 값이 클수록 vocab이 보수적으로(흔한 것만) 채워진다.
        min_frequency=MIN_FREQ,
        # special_tokens : 위에서 정의한 예약 토큰. vocab 맨 앞 인덱스(0~4)에 고정 배치된다.
        special_tokens=SPECIAL_TOKENS,
        # wordpieces_prefix : 단어 '중간/뒤' 조각에 붙이는 접두사. "개발자를"이
        #                     ["개발", "##자를"]처럼 쪼개질 때 ##가 '이건 앞 토큰에
        #                     이어지는 조각'임을 표시한다. BERT 표준 표기가 "##".
        wordpieces_prefix="##",
    )

    # -------------------------------------------------------------------------
    # [3] 저장 (OUT_DIR/vocab.txt 생성)
    # -------------------------------------------------------------------------
    tokenizer.save_model(OUT_DIR)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"[save] 토크나이저 저장 → {OUT_DIR}/  (실제 vocab={actual_vocab:,})")

    # -------------------------------------------------------------------------
    # [4] 품질 점검
    #     (a) 샘플 문장 토큰화 결과를 눈으로 확인
    #     (b) [UNK] 비율(OOV)을 코퍼스 앞부분으로 간이 측정 — 낮을수록 좋음
    # -------------------------------------------------------------------------
    print("\n[check] 샘플 토큰화 결과")
    for s in SAMPLE_TEXTS:
        toks = tokenizer.encode(s).tokens
        print(f"  · {s}")
        print(f"    → {toks}\n")

    print("[check] OOV([UNK]) 비율 간이 측정 (코퍼스 앞 5천 줄)")
    unk_id_token = "[UNK]"
    total, unk = 0, 0
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            toks = tokenizer.encode(line.strip()).tokens
            total += len(toks)
            unk += sum(1 for t in toks if t == unk_id_token)
    if total:
        print(f"  · 토큰 {total:,}개 중 [UNK] {unk:,}개 = {unk / total * 100:.3f}%")

    print("\n[done] 2단계 완료. 다음 단계 후보:")
    print("  - 도메인 어휘 점검 후, 기존 한국어 BERT 도메인 적응(continual MLM)")
    print("  - 또는 KoSPLADE/BM25 baseline 대비 검색 평가셋 구축")


if __name__ == "__main__":
    main()
