"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
from typing import Iterable
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

CHUNK_SIZE = 800        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 100      # Vì sao chọn 50? ...
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small" 
EMBEDDING_DIM = 1536

COLLECTION_NAME = "DrugLawDocs"

VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            }
        })
    if not documents:
        raise RuntimeError(f"No markdown files found in {STANDARDIZED_DIR}")
    
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    
    for doc_id, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc.get("metadata", {})

        splits = splitter.split_text(content)

        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_id": f"{metadata.get('source', 'doc')}_{i}",
                }
            })

    if not chunks:
        raise RuntimeError("Chunking resulted in 0 chunks. Check your chunking strategy and parameters.")
    
    return chunks

def _batched(items: list, batch_size: int) -> Iterable[list]:
    """Chia list thành các batch nhỏ để tránh gọi API quá lớn một lần."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
    
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    texts = [chunk["content"] for chunk in chunks]
    all_embeddings = []

    batch_size = 100  # OpenAI embedding API có giới hạn batch size, thường là 2048 tokens

    for batch_idx, batch_texts in enumerate(_batched(texts, batch_size)):
        print(f"Embedding batch {batch_idx} ({len(batch_texts)} chunks)...")
        
        response = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_MODEL
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    if len(all_embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)}, expected {len(chunks)}"
        )
    # Gán embeddings cho từng chunk
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    actual_dim = len(chunks[0]["embedding"])
    if actual_dim != EMBEDDING_DIM:
        print(
            f"⚠ Warning: EMBEDDING_DIM config = {EMBEDDING_DIM}, "
            f"but actual embedding dim = {actual_dim}."
        )

    return chunks

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.init import Auth, AdditionalConfig, Timeout
    from weaviate.classes.config import Configure, Property, DataType

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=120, insert=180)
        ),
        skip_init_checks=True,
    )
    
    try:
        # Tạo collection
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="chunk_id", data_type=DataType.TEXT)
            ]
        )
        print(f"Created collection {COLLECTION_NAME}")

        # Insert chunks
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "path": metadata.get("path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                        "chunk_id": metadata.get("chunk_id", ""),
                    },
                    vector=chunk["embedding"],
                )
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"⚠ Failed to insert {len(failed_objects)} objects.")
            print(f"First failed object: {failed_objects[0]}")
        else:
            print(f"✓ Inserted {len(chunks)} chunks into Weaviate collection {COLLECTION_NAME!r}")

    finally:
        client.close()        



def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
