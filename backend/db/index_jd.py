"""
채용공고(JD) 데이터 인덱싱 스크립트

역할:
1. data/raw/jobkorea_jd_details.csv 원본 채용공고 데이터를 파싱 및 정제합니다.
2. 수집에 성공한 채용공고를 PostgreSQL(job_posts 테이블)에 저장합니다.
3. 채용공고 본문을 RecursiveCharacterTextSplitter를 사용해 지정된 크기로 청킹(Chunking)합니다.
4. OpenAI의 text-embedding-3-small 모델을 통해 각 청크의 의미론적 벡터(1536차원)를 생성합니다.
5. 생성된 임베딩 벡터와 메타데이터(UUID point_id 기반)를 Qdrant 벡터 데이터베이스에 업로드합니다.
6. 청크 텍스트 정보와 Qdrant UUID를 PostgreSQL(job_post_chunks 테이블)에 저장하여 매핑 관계를 구성합니다.

실행 방법:
(가상환경 활성화 상태)
python -m backend.db.index_jd
"""

import os
import uuid
import json
import pandas as pd
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 공통 환경설정 로드
from backend.config import (
    POSTGRES_URL,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def load_and_clean_jd_data(csv_path):
    """
    CSV 파일을 읽어 수집 성공하고 iframe_jd_text가 존재하는 공고만 필터링/정제하여 반환합니다.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"전체 로드된 공고 수: {len(df)}")

    # 1차 필터링: status가 'success'인 성공 건수
    success_df = df[df['status'] == 'success'].copy()
    # 2차 필터링: iframe_jd_text가 존재하고 비어 있지 않은 건만 
    success_df = success_df[success_df['iframe_jd_text'].notna()]
    print(f"수집 성공 및 iframe JD 존재 공고 수: {len(success_df)}")

    cleaned_data = []

    for _, row in success_df.iterrows():
        iframe_text = str(row['iframe_jd_text']).strip()

        # json_ld 파싱 (JSON String -> dict 변환)
        json_ld_raw = row['json_ld']
        json_ld_val = json.loads(json_ld_raw)

        cleaned_data.append({
            "company": row['company'],
            "url": row['url'],
            "iframe_jd_text": iframe_text,                     # 청킹 대상 핵심 JD
            "json_ld": json_ld_val,                            # 구조화 메타데이터
        })

    print(f"최종 정제 완료된 공고 수: {len(cleaned_data)}")
    return cleaned_data

def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결 객체를 생성하여 반환합니다.
    """
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        raise

def chunk_text(text):
    """
    RecursiveCharacterTextSplitter를 사용하여 텍스트를 지정된 크기의 청크로 분할합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)

def get_embeddings(texts):
    """
    OpenAIEmbeddings를 사용하여 텍스트 리스트의 임베딩 벡터들을 생성합니다.
    """
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embeddings_model.embed_documents(texts)

def index_single_jd(conn, qdrant_client, jd):
    """
    단일 채용공고(JD)를 청킹/임베딩하여 PostgreSQL과 Qdrant에 트랜잭션 단위로 저장합니다.
    """
    cursor = conn.cursor()
    try:
        # 1. PostgreSQL: job_posts 테이블에 기본 정보 저장
        insert_post_query = """
            INSERT INTO job_posts (company, url, iframe_jd_text, json_ld, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        # dict 타입인 json_ld를 문자열로 변환하여 전송 (PostgreSQL이 알아서 JSONB로 변환)
        json_ld_str = json.dumps(jd['json_ld']) if jd['json_ld'] else None

        cursor.execute(insert_post_query, (
            jd['company'],
            jd['url'],
            jd['iframe_jd_text'],
            json_ld_str,
            'success'
        ))
        job_post_id = cursor.fetchone()[0]

        # 2. 청킹 및 임베딩 생성
        chunks = chunk_text(jd['iframe_jd_text'])
        vectors = get_embeddings(chunks)

        # 3. 청크별로 PostgreSQL Chunk 테이블 및 Qdrant 포인트 적재 준비
        qdrant_points = []
        insert_chunk_query = """
            INSERT INTO job_post_chunks (job_post_id, chunk_index, chunk_text, qdrant_point_id)
            VALUES (%s, %s, %s, %s);
        """

        for i, (chunk_text_content, vector) in enumerate(zip(chunks, vectors)):
            # 고유 UUID 생성
            point_id = str(uuid.uuid4())

            # PostgreSQL Chunk 저장
            cursor.execute(insert_chunk_query, (
                job_post_id,
                i,
                chunk_text_content,
                point_id
            ))

            # Qdrant Point 생성
            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "job_post_id": job_post_id,
                        "chunk_index": i
                    }
                )
            )

        # 4. Qdrant 업로드
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=qdrant_points
        )

        # 5. 모든 과정이 성공하면 PostgreSQL 트랜잭션 최종 커밋
        conn.commit()
        print(f"인덱싱 성공: {jd['company']} (청크 {len(chunks)}개)")
        return True

    except Exception as e:
        # 중간에 에러가 발생할 경우 PostgreSQL 롤백하여 DB 데이터 정합성 유지
        conn.rollback()
        print(f"인덱싱 실패 ({jd['company']}): {e}")
        return False
    finally:
        cursor.close()

def main():
    """
    인덱싱 스크립트의 전체 실행 흐름을 제어하는 진입점(Entry Point)입니다.
    """
    CSV_PATH = os.path.join("data", "raw", "jobkorea_jd_details.csv")
    
    print("\n[단계 1/3] CSV 데이터 로드 및 정제 시작...")
    cleaned_jds = load_and_clean_jd_data(CSV_PATH)
    
    print("\n[단계 2/3] 데이터베이스 및 Qdrant 연결 수립...")
    conn = get_db_connection()
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    print("\n[단계 3/3] 채용공고별 임베딩 생성 및 이중 데이터 적재 시작...")
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    print("\n================== 인덱싱 루프 시작 ==================")
    for jd in cleaned_jds:
        # 중복 체크
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM job_posts WHERE url = %s;", (jd['url'],))
        existing_post = cursor.fetchone()
        cursor.close()
        
        # 이미 존재하는 공고 스킵
        if existing_post:
            print(f"이미 존재하는 공고 (스킵): {jd['company']}")
            skip_count += 1
            continue
            
        # 개별 공고 인덱싱 트랜잭션 수행
        success = index_single_jd(conn, qdrant_client, jd)
        if success:
            success_count += 1
        else:
            fail_count += 1
            
    conn.close()
    print("\n================== 인덱싱 완료 ==================")
    print(f"최종 결과 -> 성공: {success_count}건, 스킵: {skip_count}건, 실패: {fail_count}건")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"인덱싱 스크립트 실행 중 에러 발생: {e}")
