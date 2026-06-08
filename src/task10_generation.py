"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

GENERATION_MODEL = "gpt-4o-mini"

MAX_CHARS_PER_CHUNK = 1800


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt dựa trên context được cung cấp.

Quy tắc bắt buộc:
- Chỉ sử dụng thông tin có trong phần Context.
- Mỗi nhận định factual phải có citation ngay sau câu hoặc mệnh đề liên quan.
- Citation phải dùng đúng nhãn nguồn trong context, ví dụ [S1], [S2].
- Không tự suy đoán hoặc bổ sung kiến thức ngoài context.
- Nếu context không đủ bằng chứng, hãy nói: "Tôi không thể xác minh thông tin này từ nguồn hiện có".
- Trả lời có cấu trúc rõ ràng, ưu tiên gạch đầu dòng khi phù hợp.
- Không cite nguồn không xuất hiện trong Context.
"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM thường chú ý tốt hơn ở đầu và cuối prompt.
    Vì vậy, ta đặt chunk tốt nhất ở đầu, chunk tốt thứ hai ở cuối,
    các chunk còn lại xen vào giữa.

    Input order theo score: [1, 2, 3, 4, 5]
    Output:                [1, 3, 5, 4, 2]

    Args:
        chunks: List sorted by score descending.

    Returns:
        List reordered để tăng khả năng LLM dùng được evidence quan trọng.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []

    # Các vị trí 0, 2, 4... đi trước: best, third-best, fifth-best...
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    # Các vị trí 1, 3... đi cuối nhưng đảo ngược:
    # với 5 chunks: thêm chunk 4 rồi chunk 2 => [1,3,5,4,2]
    start = len(chunks) - 1
    if start % 2 == 0:
        start -= 1

    for i in range(start, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def _get_source_label(chunk: dict, index: int) -> str:
    """Tạo nhãn citation ngắn gọn: S1, S2, ..."""
    return f"S{index}"


def _get_source_name(chunk: dict) -> str:
    """
    Lấy tên nguồn dễ đọc từ metadata.
    Ưu tiên source/path/filename.
    """
    metadata = chunk.get("metadata", {}) or {}

    return (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("path")
        or f"Source"
    )


def _get_doc_type(chunk: dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return metadata.get("type") or metadata.get("doc_type") or "unknown"


def _truncate_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    """Cắt chunk quá dài để tránh prompt bị phình quá mức."""
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[truncated]"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        label = _get_source_label(chunk, i)
        source_name = _get_source_name(chunk)
        doc_type = _get_doc_type(chunk)
        score = float(chunk.get("score", 0.0))
        retrieval_source = chunk.get("source", "unknown")
        metadata = chunk.get("metadata", {}) or {}

        chunk_index = metadata.get("chunk_index")
        document_id = metadata.get("document_id")

        metadata_line = (
            f"[{label}] "
            f"Source: {source_name} | "
            f"Type: {doc_type} | "
            f"Retrieval: {retrieval_source} | "
            f"Score: {score:.4f}"
        )

        if chunk_index is not None:
            metadata_line += f" | Chunk: {chunk_index}"

        if document_id:
            metadata_line += f" | Document ID: {document_id}"

        content = _truncate_text(chunk.get("content", ""))

        context_parts.append(
            f"{metadata_line}\n"
            f"{content}\n"
        )

    return "\n---\n".join(context_parts)


def _build_sources(chunks: list[dict]) -> list[dict]:
    """
    Build sources list trả về cho UI/debug.
    Gắn citation_label để biết [S1], [S2] tương ứng chunk nào.
    """
    sources = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}

        sources.append({
            "citation_label": _get_source_label(chunk, i),
            "source": _get_source_name(chunk),
            "type": _get_doc_type(chunk),
            "retrieval_source": chunk.get("source", "unknown"),
            "score": float(chunk.get("score", 0.0)),
            "chunk_index": metadata.get("chunk_index"),
            "metadata": metadata,
            "content_preview": chunk.get("content", "")[:300],
        })

    return sources


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Trên PowerShell, chạy: $env:OPENAI_API_KEY='sk-...'"
        )

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"""Context:
{context}

---

Question:
{query}

Hãy trả lời bằng tiếng Việt. Nhớ cite bằng các nhãn [S1], [S2], ... tương ứng với Context."""

    # Step 5: Call LLM
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Step 6: Return
    retrieval_sources = {
        chunk.get("source", "unknown")
        for chunk in chunks
    }

    if len(retrieval_sources) == 1:
        retrieval_source = next(iter(retrieval_sources))
    else:
        retrieval_source = "mixed"

    return {
        "answer": answer,
        # Trả sources theo thứ tự đã reorder, vì citation [S1], [S2] map theo context reorder.
        "sources": _build_sources(reordered),
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma túy?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)

        result = generate_with_citation(q)

        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")

        for source in result["sources"]:
            print(
                f"  [{source['citation_label']}] "
                f"{source['source']} | "
                f"type={source['type']} | "
                f"retrieval={source['retrieval_source']} | "
                f"score={source['score']:.4f}"
            )