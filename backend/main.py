"""
FastAPI 애플리케이션 진입점

역할:
- FastAPI 앱 초기화
- 라우터 등록
- CORS 설정
- 미들웨어 등록
- 앱 실행
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="RAG Backend API",
    description="Qdrant + PostgreSQL + LangChain RAG API",
    version="0.1.0"
)

# CORS 설정 (React 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 건강 체크 엔드포인트
@app.get("/health")
async def health_check():
    """
    서버 상태 확인 엔드포인트
    """
    return {"status": "ok"}

# 라우터 등록 (나중에 추가)
# from routes import search, upload
# app.include_router(search.router)
# app.include_router(upload.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
