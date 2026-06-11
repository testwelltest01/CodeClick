# 잡코리아 JD 커스텀 토크나이저 사전학습 코드 분석 (15만 건 데이터 기준)

> 대상 코드: `BertWordPieceTokenizer`로 JD 코퍼스에 대해 vocab 35,000짜리 WordPiece 토크나이저를 from-scratch 훈련
> 목적: Qdrant 기반 RAG(sparse + dense)에서 sparse 파트를 KoSPLADE 대신 직무 특화 어휘로 사전학습
> 데이터: `data/raw/jobkorea_jd_details_requests*.csv` **152개 파일 / 약 15만 행** (실측 기준)
> 작성일: 2026-06-11 · 개정: 15만 건 수집 반영

---

## 0. 요약 (TL;DR)

데이터가 15만 건으로 늘면서 **일부 결론은 긍정적으로 바뀌지만, 가장 중요한 한 가지는 그대로입니다.**

- ✅ **바뀐 점**: 토크나이저 학습 자체는 이제 통계적으로 정당화됩니다(vocab 35K 채울 만큼 충분). 그리고 결정적으로 **도메인 적응 사전학습(continual/DAPT) 경로가 현실적으로 열렸습니다.**
- 🔴 **그대로인 점**: **SPLADE의 sparse 표현은 "토크나이저"가 아니라 "MLM head를 가진 사전학습 모델"이 만듭니다.** 토크나이저만 새로 만드는 것으로는 sparse 모델이 생기지 않습니다. 그리고 **BERT를 from-scratch로 새로 사전학습하기에는 15만 건도 여전히 부족**합니다(아래 2-1 수치 참조).

> **권장 방향**: "새 토크나이저 + from-scratch 사전학습"이 아니라, **"코퍼스 정제 → 기존 한국어 SPLADE/BERT에 도메인 적응(continual MLM) → 필요시 도메인 vocab 확장"**. 15만 건은 정확히 이 경로에 적합한 규모입니다.

---

## 1. 실제 데이터 확인 (15만 건 실측)

`jobkorea_jd_details_requests*.csv` **152개 파일** 전체를 합쳐 측정한 결과입니다.

| 항목 | 값 |
|---|---|
| CSV 파일 수 | 152개 |
| 전체 행 수 | **148,990행** |
| `iframe_jd_text` non-null | 112,367건 |
| **중복 제거 후 고유 공고** | **92,403건** |
| 고유 공고 총 글자 수 | **약 70.2M자** |
| 공백 기준 토큰 수(고유) | **약 16.2M** |
| 평균 글자 수/공고 | 759자 |

> 📌 주의: 15만 "행"이지만 **실제 학습에 쓸 고유 텍스트는 92,403건**입니다. iframe 추출 실패(약 3.7만)·중복(약 2만)을 빼면 유효 데이터는 수집량의 약 62%입니다. 토크나이저/모델 학습은 이 92K 기준으로 계획해야 합니다.

### ⚠️ 보일러플레이트는 규모가 커져도 여전히 코퍼스를 지배함
고유 공고 92,403건에서 줄 단위 최다 반복 문구(실측):

| 반복 횟수 | 문구 |
|---|---|
| 62,749 | `포지션 및 자격요건` |
| 59,017 | `ㆍ허위사실이 발견될 경우 채용이 취소될 수 있습니다.` |
| 58,883 | `ㆍ면접일정은 추후 통보됩니다.` |
| 31,077 | `( 1명 )` |
| 26,338 | `ㆍ학력 : 학력무관` |
| 24,828 | `ㆍ상세내용을 입력하세요` |
| 24,430 | `ㆍ서류전형 > 1차면접 > 2차면접 > 임원면접 > 최종합격` |

92K 공고 중 **6만 건 이상에 동일한 정형 문구**가 들어 있습니다. 데이터가 37배 늘었어도 "직무와 무관한 쓸데없는 단어" 문제는 비율 그대로 — 오히려 절대량이 폭증했습니다. **토크나이저를 새로 만들어도 이 고빈도 문구는 vocab에 더 크게 박힙니다. 정제(cleaning/dedup)가 여전히 선행 조건입니다.**

---

## 2. AI 엔지니어 관점 분석

### 2-1. 🔴 (변함없음) 토크나이저 ≠ SPLADE. 그리고 from-scratch 사전학습엔 15만 건도 부족
SPLADE의 sparse 벡터는 **BERT의 MLM head가 각 토큰 표현을 vocab 전체로 사상하고 term importance를 부여**해 만듭니다. sparse 벡터의 차원 = 사전학습 모델 vocab 크기이고, 가중치는 사전학습된 MLM head에서 나옵니다 ([NAVER LABS – SPLADE](https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/), [SPLADE v2, Formal et al. 2021](https://www.piwowarski.fr/publication/formal_splade_2021-qabs5ajy/formal_splade_2021-QABS5AJY.pdf), [Pinecone](https://www.pinecone.io/learn/splade/)).

데이터 규모를 대입해 보면:

| 비교 | 토큰 규모 |
|---|---|
| 우리 코퍼스(고유, 공백 기준) | 약 1,600만 |
| 우리 코퍼스(WordPiece 환산 추정) | 약 3,000만~4,500만 |
| **BERT-base 원본 사전학습** | **약 33억 단어** (BookCorpus + Wikipedia) |

→ from-scratch MLM 사전학습 기준으로는 **여전히 약 100배 부족**합니다. 이 규모로 BERT를 처음부터 학습하면 underfit 되어 KoSPLADE보다 나쁜 표현이 나올 위험이 큽니다. **즉 "새 토크나이저 → from-scratch SPLADE"는 15만 건에서도 비권장입니다.**

다만 **from-scratch가 아니라 "이어 학습(continual pretraining)"이라면 이야기가 달라집니다 — 2-2 참조.**

### 2-2. ✅ (개선됨) 토크나이저 학습은 정당화 + 도메인 적응 경로가 열림
- **vocab 35K가 이제 합리적**: 공백 토큰 1,600만 규모면 `min_frequency=2`로 35,000칸을 채워도 대부분이 의미 있는 빈도의 서브워드로 채워집니다. 4천 건 시절의 "코퍼스 대비 vocab 과대" 문제는 해소되었습니다. (그래도 보일러플레이트 정제 후 24K~32K 정도가 더 깔끔합니다.)
- **도메인 적응 사전학습(DAPT)에 적합한 구간**: 수천만 토큰 도메인 코퍼스는 기존 한국어 BERT를 **이어서 MLM 학습 → SPLADE 파인튜닝**하기에 좋은 규모입니다. [Gururangan et al., *Don't Stop Pretraining* (ACL 2020)](https://aclanthology.org/2020.acl-main.740/)은 이 규모의 도메인 코퍼스로 continual pretraining 시 검색·분류 성능이 향상됨을 보였습니다. **15만 건이 진짜로 빛을 발하는 지점은 새 토크나이저가 아니라 이 도메인 적응 경로입니다.**

### 2-3. 🟠 (변함없음) 한국어에 raw WordPiece 직접 적용은 형태소 정보를 버림
`BertWordPieceTokenizer`를 한국어 원문에 바로 학습하면 "역량/역량을/역량이/역량은"이 제각각 처리됩니다. 형태소 인지(morpheme-aware) 사전 토크나이즈(Mecab/Khaiii) 후 WordPiece가 한국어 downstream에서 유의하게 낫다는 보고가 일관됩니다 ([Improving Korean NLP, arXiv:2311.03928](https://arxiv.org/pdf/2311.03928), [KR-BERT, arXiv:2008.03979](https://www.arxiv-vanity.com/papers/2008.03979/)). 데이터가 커져도 이 구조적 손실은 그대로입니다.

### 2-4. 🟡 (변함없음) 정규화/정제 누락
- `\xa0`(non-breaking space), 가운뎃점 `ㆍ`, 깨진 줄바꿈 등 잡음이 그대로 — **NFKC 정규화** 필요.
- `lowercase=True`는 **영문 약어(B2B, MD, OA, DC)**를 소문자화해 도메인 핵심 신호를 약화. 직무 검색에선 `lowercase=False` 권장.

### 2-5. 🟡 (일부 해소) 코드 레벨 버그/스멜
- **기존 코드는 `jobkorea_jd_details_requests.csv` 단일 파일만 읽음** → `glob`으로 152개 전체를 합쳐야 함. (가장 중요한 수정)
- **중복 제거 없음**: 112,367 → 고유 92,403. 보일러플레이트 반복까지 더하면 빈도 통계가 크게 편향됨. dedup + 보일러플레이트 제거 필수.
- 빈 문자열/공백만 있는 행 필터(`.str.strip()`) 필요.

### 2-6. ✅ 잘한 점
- `strip_accents=False`, `handle_chinese_chars=False`는 한국어/한자 혼용에 적절.
- 특수 토큰·`##` prefix 등 BERT 규약 준수.
- "도메인 어휘가 일반 모델 vocab에 과소대표된다"는 문제 인식은 타당 — 15만 건 규모에서는 도메인 적응으로 제대로 해결 가능.

---

## 3. 프로덕트 매니저 관점 분석

### 3-1. 목표-수단 정합성
진짜 목표는 *"토크나이저를 갖는 것"*이 아니라 *"sparse 검색이 직무 신호에 집중하게 만드는 것"*입니다. 15만 건이 모인 지금, 가장 큰 변화는 **"가설 검증 단계"에서 "실제 도메인 적응 모델을 만들 수 있는 단계"로 넘어왔다는 것**입니다. 단, 그 수단은 여전히 from-scratch 토크나이저가 아니라 도메인 적응입니다.

### 3-2. ROI / 일정 리스크 (개선됨)
- from-scratch MLM은 데이터 부족(2-1)으로 여전히 고위험·저ROI.
- **continual pretraining(DAPT)은 15만 건에서 ROI가 크게 좋아짐** — 기존 한국어 BERT 가중치를 재활용하므로 수일~1~2주 내 실험 가능, 성공 확률도 높음.
- RAG에서 sparse는 hybrid의 한 축. dense + sparse를 Qdrant에서 합치는 구조라면 **하이브리드 가중치·리랭커 튜닝**도 체감 품질을 크게 올리는 병행 과제.

### 3-3. 평가 체계 부재 (여전히 최대 리스크)
데이터가 늘어도 **"좋다"를 판정할 지표가 없으면 의사결정 불가**합니다.
- 잡코리아 도메인 **검색 평가셋**(쿼리 ↔ 관련 공고, 최소 50~100쿼리). 92K 공고가 있으니 합성 쿼리(LLM 생성) + 일부 수기 라벨로 만들기 더 쉬워짐.
- 지표: nDCG@10, Recall@50, MRR. 이게 있어야 "KoSPLADE의 쓸데없는 단어가 실제로 점수를 깎는지"를 증명 가능.

### 3-4. 유지보수·확장성
- 자체 모델/vocab은 공고 트렌드 변화마다 재학습 파이프라인 운영 부담.
- 기존 모델 기반 도메인 적응은 업스트림 개선을 흡수 가능 — 장기 유지보수가 가벼움.

---

## 4. 권장 경로 (15만 건 기준 업데이트)

> 핵심 원칙: **모델을 from-scratch로 만들지 말고, "정제 + 기존 모델 도메인 적응"으로 직무 신호에 집중시킨다.**

**1단계 — 코퍼스 정제 (즉시, 최우선)**
보일러플레이트("포지션 및 자격요건", "허위사실…", "면접일정 추후 통보", "상세내용을 입력하세요", 채용절차법 제11조 등) 룰/빈도 기반 제거 + NFKC 정규화 + 중복 제거. 92K → 깨끗한 코퍼스. **모든 후속 경로의 공통 전처리.**

**2단계 — 기존 KoSPLADE/한국어 SPLADE로 baseline + 평가셋**
정제 코퍼스로 인덱싱하고 3-3 평가셋으로 nDCG/Recall 측정. "쓸데없는 단어"가 실제 점수를 깎는지 데이터로 확인.

**3단계 — 도메인 적응 (15만 건이 빛나는 지점)**
- **continual MLM pretraining(DAPT)**: 기존 한국어 BERT를 정제 코퍼스(약 70M자)로 **이어서** MLM 학습 → SPLADE 파인튜닝. from-scratch가 아니라 이어 학습이라 이 규모에 적합 ([Gururangan et al. 2020](https://aclanthology.org/2020.acl-main.740/)).
- **도메인 vocab 확장**: 직무 용어만 골라 기존 vocab에 추가 — [exBERT (EMNLP 2020)](https://aclanthology.org/2020.findings-emnlp.129/), [AVocaDo (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.385.pdf). from-scratch보다 검증되고 적은 자원으로 "vocab mismatch 해소"라는 목적을 그대로 달성.
- **출력 vocab 제어/가지치기**: SPLADE는 vocab 차원 위 가중치이므로, 도메인 불용어·법령어 출력 차원을 마스킹/감쇠하면 학습 없이 "쓸데없는 단어"를 억제.

**4단계 — (선택) 더 단순한 sparse: BM25/BM42 + 사용자 사전**
Mecab 사용자 사전에 직무 용어 등록한 BM25/BM42(Qdrant 지원)는 학습 없이도 강력·해석가능. 평가셋에서 KoSPLADE가 BM25 대비 이득이 없다면 이 단순 경로가 정답일 수 있음.

**from-scratch 토크나이저 + 사전학습이 정당화되는 시점?** 코퍼스가 수억~수십억 토큰으로 늘고(현재 약 1,600만~4,500만 토큰), 도메인 적응(3단계)으로도 한계가 확인될 때. **15만 건(고유 92K)은 도메인 적응의 적기이지, from-scratch 사전학습의 적기는 아직 아님.**

---

## 5. 그래도 토크나이저를 학습한다면 — 최소 수정 코드

(탐색/분석 목적 한정. 152개 파일 전체 로드 + 정제 + 코퍼스 규모에 맞춘 설정)

```python
import glob, unicodedata
import pandas as pd
from tokenizers import BertWordPieceTokenizer

# 1) 152개 CSV 전체 로드 + 고유화
files = sorted(glob.glob("data/raw/jobkorea_jd_details_requests*.csv"))
texts = []
for fp in files:
    df = pd.read_csv(fp)
    if "iframe_jd_text" in df.columns:
        texts += df["iframe_jd_text"].dropna().astype(str).tolist()
texts = [t for t in dict.fromkeys(texts) if t.strip()]   # 중복/공백 제거 → 약 92K

# 2) NFKC 정규화 + 보일러플레이트 라인 제거
BOILER = {
    "포지션 및 자격요건", "ㆍ면접일정은 추후 통보됩니다.",
    "ㆍ허위사실이 발견될 경우 채용이 취소될 수 있습니다.",
    "ㆍ상세내용을 입력하세요", "ㆍ학력 : 학력무관", "ㆍ경력 : 경력무관",
    "ㆍ서류전형 > 1차면접 > 2차면접 > 임원면접 > 최종합격",
}
def clean(t):
    t = unicodedata.normalize("NFKC", t).replace("\xa0", " ")
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln and ln not in BOILER]
    return "\n".join(lines)

with open("jd_corpus.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(clean(t) + "\n")

# 3) 15만 건 규모에 맞춘 설정 (vocab 24K~32K, 약어 보존)
tok = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=False,
    strip_accents=False, lowercase=False,   # B2B, MD 등 영문 약어 보존
)
tok.train(
    files=["jd_corpus.txt"],
    vocab_size=32000,        # 코퍼스 규모상 35K도 가능하나 정제 후 32K가 깔끔
    min_frequency=3,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    wordpieces_prefix="##",
)
tok.save_model("custom_job_tokenizer")
```
> ⚠️ 이 토크나이저만으로는 SPLADE가 만들어지지 않습니다. sparse 표현을 얻으려면 이 vocab으로 MLM을 학습해야 하는데, **15만 건은 from-scratch엔 부족**합니다. 실제 제품에는 4장의 **도메인 적응(continual pretraining)** 경로를 사용하세요.

---

## 참고 문헌
- Formal et al., **SPLADE v2: Sparse Lexical and Expansion Model for IR** (2021) — [PDF](https://www.piwowarski.fr/publication/formal_splade_2021-qabs5ajy/formal_splade_2021-QABS5AJY.pdf)
- NAVER LABS Europe, **SPLADE: a sparse bi-encoder BERT-based model** — [blog](https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/)
- Pinecone, **SPLADE for Sparse Vector Search Explained** — [link](https://www.pinecone.io/learn/splade/)
- Gururangan et al., **Don't Stop Pretraining: Adapt Language Models to Domains and Tasks**, ACL 2020 — [ACL](https://aclanthology.org/2020.acl-main.740/)
- **Improving Korean NLP Tasks with Linguistically Informed Subword Tokenization and Sub-character Decomposition**, arXiv:2311.03928 — [PDF](https://arxiv.org/pdf/2311.03928)
- **KR-BERT: A Small-Scale Korean-Specific Language Model**, arXiv:2008.03979 — [link](https://www.arxiv-vanity.com/papers/2008.03979/)
- Tai et al., **exBERT: Extending Pre-trained Models with Domain-specific Vocabulary Under Constrained Training Resources**, Findings of EMNLP 2020 — [ACL](https://aclanthology.org/2020.findings-emnlp.129/)
- Hong et al., **AVocaDo: Strategy for Adapting Vocabulary to Downstream Domain**, EMNLP 2021 — [PDF](https://aclanthology.org/2021.emnlp-main.385.pdf)
