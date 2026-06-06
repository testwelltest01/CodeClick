-- JD 테이블: data\raw\jobkorea_jd_details.csv 와 대응된다.
CREATE TABLE job_posts (
    id SERIAL PRIMARY KEY,
    company VARCHAR(255) NOT NULL,
    url TEXT NOT NULL UNIQUE,
    iframe_jd_text TEXT,
    main_jd_text TEXT,
    json_ld JSONB,
    status VARCHAR(50),
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- chunk 테이블: 공고를 여러 조각으로 나누어 저장
CREATE TABLE job_post_chunks (
    id SERIAL PRIMARY KEY,
    job_post_id INTEGER NOT NULL REFERENCES job_posts(id) ON DELETE CASCADE,
    -- REFERENCES job_posts(id): job_posts 테이블의 id를 가리킨다
    -- ON DELETE CASCADE: 부모 공고가 삭제되면, 그 공고에 연결된 chunk들도 자동 삭제된다
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    qdrant_point_id UUID NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_post_id, chunk_index) -- job_post_id와 chunk_index의 조합이 유일해야 한다
);

-- 인덱스 생성: 검색 성능 향상을 위해 url, job_post_id, qdrant_point_id에 인덱스를 추가
CREATE INDEX idx_job_post_url ON job_posts(url);
CREATE INDEX idx_job_post_chunks_job_post_id ON job_post_chunks(job_post_id);
CREATE INDEX idx_job_post_chunks_qdrant_point_id ON job_post_chunks(qdrant_point_id);