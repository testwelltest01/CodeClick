# CodeClick

잡코리아 채용공고 데이터를 수집해 JD 기반 Vector DB를 만들고, 사용자의 자기소개서와 유사한 채용공고를 추천/분석하는 RAG 기반 프로젝트입니다.

현재 구성은 다음과 같습니다.

```text
FastAPI backend
PostgreSQL metadata/raw data DB
Qdrant vector DB
React frontend
OpenAI embedding API
```

## 프로젝트 구조

```text
CodeClick/
  backend/
    db/
      schema.sql
      init_qdrant.py
    config.py
    main.py
    requirements.txt
  data/
    raw/
  frontend/
  scripts/
  docker-compose.yml
  .env.example
```

## 사전 준비

공통으로 아래 도구가 필요합니다.

- Python 3.10 이상 권장
- Docker Desktop
- Git
- Node.js, npm
- OpenAI API key

`.env` 파일은 Git에 올리지 않습니다. 처음 실행하는 로컬에서는 `.env.example`을 복사해서 `.env`를 직접 만듭니다.

## Windows 환경 준비

PowerShell 기준입니다.

1. 레포지토리 클론

```powershell
git clone <repo-url>
cd CodeClick
```

2. 환경변수 파일 생성

```powershell
Copy-Item .env.example .env
```

생성된 `.env`에서 `OPENAI_API_KEY` 값을 본인 키로 수정합니다.

3. Python 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

만약 PowerShell 실행 정책 때문에 활성화가 막히면 현재 터미널 세션에서만 다음 명령을 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

4. 백엔드 의존성 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

5. Docker DB 실행

```powershell
docker compose up -d
docker compose ps
```

`postgres`, `qdrant` 컨테이너가 `Up` 상태면 정상입니다.

6. PostgreSQL 테이블 생성

```powershell
Get-Content backend\db\schema.sql | docker exec -i postgres psql -U rag_user -d rag_db
```

7. Qdrant collection 초기화

```powershell
python -m backend.db.init_qdrant
```

8. FastAPI 실행

```powershell
python -m uvicorn backend.main:app --reload
```

브라우저에서 확인합니다.

```text
http://localhost:8000/health
http://localhost:6333/collections
```

## macOS 환경 준비

zsh 또는 bash 기준입니다.

1. 레포지토리 클론

```bash
git clone <repo-url>
cd CodeClick
```

2. 환경변수 파일 생성

```bash
cp .env.example .env
```

생성된 `.env`에서 `OPENAI_API_KEY` 값을 본인 키로 수정합니다.

3. Python 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. 백엔드 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

5. Docker DB 실행

```bash
docker compose up -d
docker compose ps
```

`postgres`, `qdrant` 컨테이너가 `Up` 상태면 정상입니다.

6. PostgreSQL 테이블 생성

```bash
docker exec -i postgres psql -U rag_user -d rag_db < backend/db/schema.sql
```

7. Qdrant collection 초기화

```bash
python -m backend.db.init_qdrant
```

8. FastAPI 실행

```bash
python -m uvicorn backend.main:app --reload
```

브라우저에서 확인합니다.

```text
http://localhost:8000/health
http://localhost:6333/collections
```

## Frontend 실행

프론트엔드는 `frontend/` 디렉터리에서 실행합니다.

```bash
cd frontend
npm install
npm run dev
```

기본 개발 서버 주소는 보통 다음과 같습니다.

```text
http://localhost:5173
```

## 주의사항

- `.env`는 개인 API key가 들어가므로 Git에 커밋하지 않습니다.
- `.venv/`, `postgres_data/`, `qdrant_storage/`는 로컬 생성물이라 Git에 커밋하지 않습니다.
- `schema.sql`과 `init_qdrant.py`는 초기화용 코드입니다. 이미 생성된 DB/table/collection에 다시 실행하면 중복 생성 오류가 날 수 있습니다.
- Qdrant collection 이름은 `job_descriptions`입니다.
- OpenAI embedding model은 `text-embedding-3-small`, vector dimension은 `1536`입니다.
