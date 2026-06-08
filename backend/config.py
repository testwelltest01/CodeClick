"""
환경설정 파일

역할:
- 환경변수 관리
- 데이터베이스 연결 설정
- API 키 관리
- 앱 설정값 중앙화
"""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant 벡터 DB 설정
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = "job_descriptions"

# PostgreSQL 메타데이터 DB 설정
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# LangChain 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding
# text-embedding-3-small의 embedding 차원 수
EMBEDDING_DIMENSION = 1536

# RAG 검색 설정
SEARCH_LIMIT = 5  # 상위 5개 문서 반환
CHUNK_SIZE = 1000  # 텍스트 청킹 크기
CHUNK_OVERLAP = 200  # 청크 겹침

# 앱 설정
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
