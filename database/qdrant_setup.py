from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

COLLECTION_NAME = "notes_chunks"


def main() -> None:
    client = QdrantClient(host="localhost", port=6333)

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
