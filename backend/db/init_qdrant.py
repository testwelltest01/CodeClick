from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


def init_qdrant_collection():
    # Qdrant 서버에 연결합니다.
    client = QdrantClient(url=QDRANT_URL)

    # 임베딩 벡터를 저장할 컬렉션을 생성합니다.
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        )
    )

    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' initialized with dimension {EMBEDDING_DIMENSION}.")

if __name__ == "__main__":
    init_qdrant_collection()
    
