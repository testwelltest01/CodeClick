from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


def init_qdrant_collection():
    client = QdrantClient(url=QDRANT_URL)

    # 컬렉션 설정 변경 시 기존 컬렉션 재구축이 필요하므로 삭제를 시도합니다.
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Existing collection '{QDRANT_COLLECTION_NAME}' deleted.")
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            "text-sparse": SparseVectorParams()
        }
    )

    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' initialized with dimension {EMBEDDING_DIMENSION} and sparse vectors.")

if __name__ == "__main__":
    init_qdrant_collection()
    