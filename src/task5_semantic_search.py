"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from typing import Any

import weaviate
from openai import OpenAI
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery


# =============================================================================
# CONFIGURATION
# =============================================================================

COLLECTION_NAME = "DrugLawDocs"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

DEFAULT_TOP_K = 10


# =============================================================================
# HELPERS
# =============================================================================

def get_openai_client() -> OpenAI:
    """Khởi tạo OpenAI client từ biến môi trường OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY")

    return OpenAI(api_key=api_key)


def embed_query(query: str) -> list[float]:
    """Embed query bằng OpenAI embedding model."""
    query = query.strip()

    client = get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    embedding = response.data[0].embedding

    if len(embedding) != EMBEDDING_DIM:
        print(
            f"⚠ Warning: expected embedding dim {EMBEDDING_DIM}, "
            f"but got {len(embedding)}"
        )

    return embedding


def get_weaviate_client():
    """
    Kết nối Weaviate Cloud.

    Cần set:
        WEAVIATE_URL
        WEAVIATE_API_KEY
    """
    cluster_url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=120, insert=180)
        ),
        skip_init_checks=True,
    )


def format_result(obj: Any) -> dict:
    """
    Chuẩn hóa object trả về từ Weaviate thành format dùng cho các task sau.

    Returns:
        {
            "content": str,
            "score": float,
            "metadata": dict
        }
    """
    props = obj.properties or {}

    # Với near_vector, Weaviate thường trả distance.
    # Distance càng nhỏ càng gần. Ta đổi thành score = 1 - distance để dễ hiểu.
    distance = None
    score = None

    if obj.metadata:
        distance = getattr(obj.metadata, "distance", None)

    if distance is not None:
        score = 1.0 - float(distance)
    else:
        score = 0.0

    return {
        "content": props.get("content", ""),
        "score": score,
        "metadata": {
            "source": props.get("source", ""),
            "path": props.get("path", ""),
            "type": props.get("doc_type", ""),
            "chunk_index": props.get("chunk_index", None),
            "chunk_id": props.get("chunk_id", ""),
            "distance": distance,
        },
    }

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model ở Task 4
    # Bước 2: Query vector store (cosine similarity)
    # Bước 3: Return top_k results

    query_vector = embed_query(query)

    client = get_weaviate_client()

    try:
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[
                "content",
                "source",
                "path",
                "doc_type",
                "chunk_index",
                "chunk_id",
            ],
        )
        results = [format_result(obj) for obj in response.objects]

        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]

        print("=" * 80)
        print(f"Rank {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Distance: {metadata.get('distance')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Type: {metadata.get('type')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print("-" * 80)
        print(result["content"][:700].replace("\n", " "))
        print()
